#include "utils/utils.hpp"

namespace chaos
{
	ObjectRect::ObjectRect(const Rect& rect, float score) : rect(rect), score(score) {}

	std::vector<int> SoftNMS(std::vector<ObjectRect>& objects, double overlap_rate, double min_confidence, IoUType iou_type, WeightType weight_type)
	{
		std::multimap<float, int> score_mapper;
		for (int i = 0; i < objects.size(); i++)
		{
			score_mapper.insert(std::multimap<float, int>::value_type(objects[i].score, (int)i));
		}

		std::vector<int> picked;
		while (!score_mapper.empty())
		{
			auto last_item = score_mapper.rbegin();
			int last_idx = score_mapper.rbegin()->second; // get the index of maximum score value

			picked.push_back(last_idx);
			for (auto it = score_mapper.begin(); it != score_mapper.end();)
			{
				int idx = it->second;
				if (idx == last_idx)
				{
					auto next_item = it;
					next_item++;
					score_mapper.erase(it);
					it = next_item;
					continue;
				}

				Rect overlap = objects[idx].rect & objects[last_idx].rect;

				float overlap_value;
				switch (iou_type)
				{
				case IOU_MAX:
					overlap_value = overlap.area() / std::max(objects[idx].rect.area(), objects[last_idx].rect.area());
					break;
				case IOU_MIN:
					overlap_value = overlap.area() / std::min(objects[idx].rect.area(), objects[last_idx].rect.area());
					break;
				case IOU_UNION:
				default:
					overlap_value = overlap.area() / (objects[idx].rect.area() + objects[last_idx].rect.area() - overlap.area());
					break;
				}

				float weight = 1.0f;
				switch (weight_type)
				{
				case WEIGHT_LINEAR:
					weight = overlap_value > overlap_rate ? 1.f - overlap_value : 1.f;
					break;
				case WEIGHT_GAUSSIAN:
					weight = exp((-overlap_value * overlap_value) / 0.5f);
					break;
				case WEIGHT_ORIGINAL:
				default:
					weight = overlap_value > overlap_rate ? 0.f : 1.f;
					break;
				}
				objects[idx].score *= weight;

				if (objects[idx].score < min_confidence)
				{
					auto next_item = it;
					next_item++;
					score_mapper.erase(it);
					it = next_item;
				}
				else
				{
					it++;
				}
			}
		}
		return picked;
	}

	void MakeRectSquare(Rect& rect)
	{
		auto max_side = std::max(rect.width, rect.height);
		Point tl{ rect.x + (rect.width / 2.f - max_side / 2.f), rect.y + (rect.height / 2.f - max_side / 2.f) };
		Size size{ max_side, max_side };

		rect = Rect(tl, size);
	}

	Mat LineSpace(float a, float b, int n)
	{
		CHECK_LE(a, b);
		CHECK_GT(n, 1);

		float step = (b - a) / (n - 1);
		Mat space(1, n, CV_32FC1);

		auto ptr = space.ptr<float>();
		for (int i = 0; i < n; i++)
		{
			ptr[i] = a + i * step;
		}

		return space;
	}

	Mat Crop(const cv::Mat& src, const Rect& roi, const Size& size, int flags, int border_type, const Scalar& border_value)
	{
		Mat m = (cv::Mat_<float>(2, 3) <<
			size.width / roi.width, 0, -roi.x * size.width / roi.width,
			0, size.height / roi.height, -roi.y * size.height / roi.height);
		Mat cropped;
		warpAffine(src, cropped, m, size, flags, border_type, border_value);
		return cropped;
	}

	Mat FindNonReflectiveTransform(std::vector<Point> source_points, std::vector<Point> target_points, Mat& T_inv)
	{
		CHECK_EQ(source_points.size(), target_points.size());
		CHECK_GE(source_points.size(), 2);
		Mat U = Mat::zeros((int)target_points.size() * 2, 1, CV_32F);
		Mat X = Mat::zeros((int)source_points.size() * 2, 4, CV_32F);
		for (int i = 0; i < target_points.size(); i++) {
			U.at<float>(i * 2, 0) = source_points[i].x;
			U.at<float>(i * 2 + 1, 0) = source_points[i].y;
			X.at<float>(i * 2, 0) = target_points[i].x;
			X.at<float>(i * 2, 1) = target_points[i].y;
			X.at<float>(i * 2, 2) = 1;
			X.at<float>(i * 2, 3) = 0;
			X.at<float>(i * 2 + 1, 0) = target_points[i].y;
			X.at<float>(i * 2 + 1, 1) = -target_points[i].x;
			X.at<float>(i * 2 + 1, 2) = 0;
			X.at<float>(i * 2 + 1, 3) = 1;
		}
		Mat R = X.inv(cv::DECOMP_SVD) * U;
		T_inv = (cv::Mat_<float>(3, 3) <<
			R.at<float>(0), -R.at<float>(1), 0,
			R.at<float>(1), R.at<float>(0), 0,
			R.at<float>(2), R.at<float>(3), 1);

		Mat T = T_inv.inv(cv::DECOMP_SVD);
		T_inv = T_inv(Rect(0, 0, 2, 3)).t();
		return T(Rect(0, 0, 2, 3)).t();
	}

	Mat FindSimilarityTransform(std::vector<Point> source_points, std::vector<Point> target_points, Mat& T_inv)
	{
		Mat Tinv1, Tinv2;
		Mat trans1 = FindNonReflectiveTransform(source_points, target_points, Tinv1);
		std::vector<Point> source_point_reflect;
		for (auto sp : source_points)
		{
			source_point_reflect.push_back(Point(-sp.x, sp.y));
		}
		Mat trans2 = FindNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
		trans2.colRange(0, 1) *= -1;
		std::vector<Point> trans_points1, trans_points2;
		transform(source_points, trans_points1, trans1);
		transform(source_points, trans_points2, trans2);
		double norm1 = cv::norm(Mat(trans_points1), Mat(target_points), cv::NORM_L2);
		double norm2 = cv::norm(Mat(trans_points2), Mat(target_points), cv::NORM_L2);
		T_inv = norm1 < norm2 ? Tinv1 : Tinv2;
		return norm1 < norm2 ? trans1 : trans2;
	}
}