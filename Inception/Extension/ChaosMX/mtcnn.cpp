#include "face/detector.hpp"
#include "dnn/group.hpp"

namespace chaos
{
	namespace face
	{
		class MultiTaskCNN : public Detector
		{
		public:
			MultiTaskCNN(const std::string& folder, const dnn::Context& ctx)
			{
				std::vector<std::string> names = { "PNet", "RNet", "ONet" };
				std::map<std::string, dnn::DataLayer> inputs = {
					{"PNet", {"data", {1,3,12,12}}},
					{"RNet", {"data", {1,3,24,24}}},
					{"ONet", {"data", {1,3,48,48}}}
				};
				for (auto name : names)
				{
					File symbol = folder + "\\" + name + ".json";
					File weight = folder + "\\" + name + ".params";
					dnn::GroupNet::Load({ symbol, weight }, ctx).As(name).InTo(nets);

					nets[name]->BindExecutor({ inputs[name] });
				}

				nets.SetForward("PNet", [&]() { PNetForward(); })
					.SetForward("RNet", [&]() { PNetForward(); })
					.SetForward("ONet", [&]() { ONetForward(); });
			}

			~MultiTaskCNN() {}

			std::vector<FaceInfo> Detect(const Mat& image) final
			{
				//CHECK(image.rows > 12 && image.cols > 12);
				if (image.rows < 12 || image.cols < 12)
				{
					return std::vector<FaceInfo>();
				}

				// Pre-process
				data = image.t();
				data.convertTo(data, CV_32F, 1 / 128., -1.);

				std::vector<float>().swap(scales);
				float scale = 12.f / min_face;
				while (floor(image.cols * (double)scale * scale_decay >= 12 && floor(image.rows * (double)scale * scale_decay) >= 12))
				{
					scales.push_back(scale);
					scale *= (float)scale_decay;
				}

				// Forward
				nets.Forward("PNet").Forward("RNet").Forward("ONet");

				// Post-process
				std::vector<FaceInfo> faces_info(objects.size());
				for (size_t i = 0; i < objects.size(); i++)
				{
					faces_info[i] = objects[i];
					if (do_landmark)
					{
						faces_info[i].points = landmarks[i];
					}
				}

				return faces_info;
			}

			void Detect(const Mat& image, FaceInfo& info) final
			{
				//CHECK(image.rows > 12 && image.cols > 12);
				if (image.rows < 12 || image.cols < 12)
				{
					return;
				}

				data = image.t();
				data.convertTo(data, CV_32F, 1 / 128., -1.);

				std::vector<ObjectRect>().swap(objects);
				// Transpose the rect
				objects.push_back({ Rect(info.rect.y, info.rect.x, info.rect.height, info.rect.width), info.score });

				nets.Forward("ONet");

				if (!objects.empty())
				{
					info = objects[0];
					if (do_landmark)
					{
						info.points = landmarks[0];
					}
				}
			}

		private:
			void Parse(const std::any& any) final
			{
				if (any.type() == typeid(const char*) && args_list.find(std::any_cast<const char*>(any)) != args_list.end())
				{
					const char* arg = std::any_cast<const char*>(any);
					try
					{
						switch (Hash(arg))
						{
						case "SacaleDecay"_hash:
							scale_decay = std::any_cast<double>(arg_value);
							break;
						case "MinFace"_hash:
							min_face = std::any_cast<int>(arg_value);
							break;
						case "NMS"_hash:
							nms_threshold = std::any_cast<double>(arg_value);
							break;
						case "DoLandmark"_hash:
							do_landmark = std::any_cast<bool>(arg_value);
							break;
						case "Confidence"_hash:
							confidence = std::any_cast<std::vector<double>>(arg_value);
							break;
						default:
							LOG(WARNING) << "Unknown arg " << arg;
							break;
						}
					}
					catch (std::bad_any_cast err)
					{
						LOG(FATAL) << arg << " cast error " << err.what();
					}
				}
				else
				{
					arg_value = any;
				}
			}

			void PNetForward()
			{
				std::vector<ObjectRect> results;
				for (auto s : scales)
				{
					cv::Mat input;
					cv::resize(data, input, cv::Size(), s, s);

					dnn::Tensor prob, bounding;
					nets["PNet"]->Reshape({ {"data", {1,3, input.rows, input.cols}} });
					nets["PNet"]->SetLayerData("data", dnn::Tensor::Unroll({ input }));
					nets["PNet"]->Forward();
					nets["PNet"]->GetLayerData("conv4_1_output", prob); // 1x2xhxw
					nets["PNet"]->GetLayerData("conv4_2_output", bounding); // 1x4xhxw

					std::vector<ObjectRect> scale_results;
					int rows = prob.shape[2], cols = prob.shape[3];

					for (int r = 0; r < rows; r++)
					{
						for (int c = 0; c < cols; c++)
						{
							//// Softmax
							float fg = prob.At<float>({ 0, 1, r, c });
							float bg = prob.At<float>({ 0, 0, r, c });
							float score = exp(fg) / (exp(fg) + exp(bg));
							if (score > confidence[0])
							{
								float x = c * 2.f;
								float y = r * 2.f;
								float w = 12.f;
								float h = 12.f;

								x += 12.f * bounding.At<float>({ 0, 1, r, c });
								y += 12.f * bounding.At<float>({ 0, 0, r, c });
								w += 12.f * (bounding.At<float>({ 0, 3, r, c }) - bounding.At<float>({ 0, 1, r, c }));
								h += 12.f * (bounding.At<float>({ 0, 2, r, c }) - bounding.At<float>({ 0, 0, r, c }));
								Rect rect(x / s, y / s, w / s, h / s);
								if (rect.width >= 12 && rect.height >= 12)
								{
									scale_results.push_back({ rect, score });
								}
							}
						}
					}

					auto picked = SoftNMS(scale_results, nms_threshold, confidence[0]);
					for (auto p : picked)
					{
						results.push_back(scale_results[p]);
					}
				}

				std::vector<ObjectRect>().swap(objects);
				auto picked = SoftNMS(results, nms_threshold, confidence[0]);
				for (auto p : picked)
				{
					objects.push_back(results[p]);
				}
			}

			void RNetForward()
			{
				if (objects.empty()) return;

				std::vector<ObjectRect> results;

				nets["RNet"]->Reshape({ {"data", {(int)objects.size(), 3, 24, 24}} });
				std::vector<cv::Mat> input;
				for (auto& obj : objects)
				{
					MakeRectSquare(obj.rect);
					input.push_back(Crop(data, obj.rect, cv::Size(24, 24)));
				}

				dnn::Tensor prob, bounding;
				nets["RNet"]->SetLayerData("data", dnn::Tensor::Unroll({ input }));
				nets["RNet"]->Forward();
				nets["RNet"]->GetLayerData("conv5_1_output", prob);
				nets["RNet"]->GetLayerData("conv5_2_output", bounding);

				for (int i = 0; i < prob.shape[0]; i++)
				{
					auto bg_prob = ((float*)prob.data)[i * 2];
					auto fc_prob = ((float*)prob.data)[i * 2 + 1];
					float score = exp(fc_prob) / (exp(bg_prob) + exp(fc_prob));

					float* rect_ptr = ((float*)bounding.data) + i * (size_t)4;
					if (score > confidence[1])
					{
						auto y = objects[i].rect.y + objects[i].rect.height * rect_ptr[0];
						auto x = objects[i].rect.x + objects[i].rect.width * rect_ptr[1];
						auto height = objects[i].rect.height * (1 + rect_ptr[2] - rect_ptr[0]);
						auto width = objects[i].rect.width * (1 + rect_ptr[3] - rect_ptr[1]);
						Rect this_rect{ x, y, width, height };
						if (width >= 12 && height >= 12)
							results.push_back({ this_rect, score });

					}
				}

				std::vector<ObjectRect>().swap(objects);
				auto picked = SoftNMS(results, nms_threshold, confidence[1]);
				for (auto p : picked)
				{
					objects.push_back(results[p]);
				}
			}

			void ONetForward()
			{
				if (objects.empty()) return;

				std::vector<ObjectRect> results;
				std::vector<Landmark> all_points;

				nets["ONet"]->Reshape({ {"data", {(int)objects.size(), 3, 48, 48}} });
				std::vector<Mat> input;
				for (auto& obj : objects)
				{
					MakeRectSquare(obj.rect);
					input.push_back(Crop(data, obj.rect, cv::Size(48, 48)));
				}

				dnn::Tensor prob, bounding, points;
				nets["ONet"]->SetLayerData("data", dnn::Tensor::Unroll({ input }));
				nets["ONet"]->Forward();
				nets["ONet"]->GetLayerData("conv6_1_output", prob);
				nets["ONet"]->GetLayerData("conv6_2_output", bounding);
				nets["ONet"]->GetLayerData("conv6_3_output", points);

				for (int i = 0; i < prob.shape[0]; i++)
				{
					auto bg_prob = ((float*)prob.data)[i * 2];
					auto fc_prob = ((float*)prob.data)[i * 2 + 1];
					float score = exp(fc_prob) / (exp(bg_prob) + exp(fc_prob));

					float* rect_ptr = ((float*)bounding.data) + i * (size_t)4;
					float* points_ptr = ((float*)points.data) + i * (size_t)10;

					if (score > confidence[2])
					{
						auto y = objects[i].rect.y + objects[i].rect.height * rect_ptr[0];
						auto x = objects[i].rect.x + objects[i].rect.width * rect_ptr[1];
						auto height = objects[i].rect.height * (1 + rect_ptr[2] - rect_ptr[0]);
						auto width = objects[i].rect.width * (1 + rect_ptr[3] - rect_ptr[1]);

						Rect this_rect{ y, x, height, width }; //{ x, y, width, height }; // Transpose
						if (width >= 12 && height >= 12)
						{
							results.push_back({ this_rect, score });
							if (do_landmark)
							{
								Landmark pts;
								for (int p = 0; p < 5; p++)
								{
									// Transpose
									pts.push_back(Point(points_ptr[p] * objects[i].rect.height + objects[i].rect.y,
										points_ptr[p + 5] * objects[i].rect.width + objects[i].rect.x));
								}
								all_points.push_back(pts);
							}
						}
					}
				}

				std::vector<ObjectRect>().swap(objects);
				std::vector<Landmark>().swap(landmarks);
				auto picked = SoftNMS(results, nms_threshold, confidence[2], IOU_MIN);
				for (auto p : picked)
				{
					objects.push_back(results[p]);
					if (do_landmark) landmarks.push_back(all_points[p]);
				}
			}

			std::set<std::string> args_list = { "ScaleDecay", "MinFace", "NMS", "DoLandmark", "Confidence" };

			int min_face = 40;
			double scale_decay = 0.709;
			std::vector<double> confidence = { 0.5, 0.7, 0.7 };
			double nms_threshold = 0.5;
			bool do_landmark = true;

			dnn::GroupNet nets;

			Mat data;
			std::vector<ObjectRect> objects;
			std::vector<Landmark> landmarks;
			std::vector<float> scales;

		};

		Ptr<Detector> Detector::LoadMTCNN(const std::string& folder, const dnn::Context& ctx)
		{
			return Ptr<Detector>(new MultiTaskCNN(folder, ctx));
		}
	}
}