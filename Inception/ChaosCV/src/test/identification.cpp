#include "test/test_engine.hpp"
#include "utils/utils.hpp"

#include <rocksdb/db.h>

#include <fstream>

namespace chaos
{
	namespace test
	{
		inline float Interpolate(const Mat& X, const Mat& Y, float x)
		{
			CHECK_EQ(1, X.rows);
			CHECK_EQ(1, Y.rows);

			double min_value;
			int min_idx[2];
			cv::minMaxIdx(cv::abs(X - x), &min_value, nullptr, min_idx);
			float x1 = X.at<float>(min_idx);
			float y1 = Y.at<float>(min_idx);

			int o = x - x1 > 0 ? 1 : -1;
			float x2 = X.at<float>(0, min_idx[1] + o);
			float y2 = Y.at<float>(0, min_idx[1] + o);

			float y = (y2 - y1) / (x2 - x1 + FLT_EPSILON) * (x - x1) + y1;
			return y;
		}

		ITest::~ITest() {}

		void ITest::SetGallery(const Ptr<DataLoader>& loader) { gallery = loader; }
		void ITest::SetGenuine(const Ptr<DataLoader>& loader) { genuine = loader; }
		void ITest::SetMeasure(const std::function<double(const Mat&, const Mat&)>& func) { measure = func; }

		CumulativeTabel ITest::GetCumulative() const { return cumulative; }
		ConfusionMat ITest::GetConfusion() const { return confusion; }
		ConfusionTable ITest::GetGlobalConfusion() const { return global_confusion; }
		std::vector<ConfusionTable> ITest::GetLocalConfusions() const { return local_confusions; }


		class Identification : public ITest
		{
		public:
			Identification() : database(nullptr) {}
			~Identification()
			{
				Close();
			}

			void Create(const std::string& db)
			{
				rocksdb::Options options;
				options.create_if_missing = true;
				options.error_if_exists = true;

				std::vector<rocksdb::ColumnFamilyDescriptor> descriptors = {
					rocksdb::ColumnFamilyDescriptor("Gallery", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor("Genuine", rocksdb::ColumnFamilyOptions()),
				};

				status = rocksdb::DB::Open(options, db, &database);
				CHECK(status.ok()) << status.ToString();
				status = database->CreateColumnFamilies(descriptors, &handles);
				CHECK(status.ok()) << status.ToString();

				folder = db;
				can_run = true;
			}

			void Load(const std::string& db)
			{
				can_run = false;
				folder = db;
				
				rocksdb::Options options;
				options.create_if_missing = false;
				options.error_if_exists = false;

				std::vector<rocksdb::ColumnFamilyDescriptor> descriptors = {
					rocksdb::ColumnFamilyDescriptor("Gallery", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor("Genuine", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName, rocksdb::ColumnFamilyOptions()),
				};
				
				status = rocksdb::DB::Open(options, db, descriptors, &handles, &database);
				CHECK(status.ok()) << status.ToString();

				std::string buff;
				status = database->Get(rocksdb::ReadOptions(), "NOC", &buff);
				CHECK(status.ok()) << status.ToString();
				noc = std::stoll(buff);

				status = database->Get(rocksdb::ReadOptions(), "Genuine Label", &buff);
				CHECK(status.ok()) << status.ToString();
				size_t n = buff.size() / sizeof(int);
				for (int i = 0; i < n; i++)
				{
					int label = ((int*)buff.data())[i];
					genuine_label.insert(label);
				}

				status = database->Get(rocksdb::ReadOptions(), "Valid Size", &buff);
				CHECK(status.ok()) << status.ToString();
				valid_size[0] = ((int*)buff.data())[0];
				valid_size[1] = ((int*)buff.data())[1];

				status = database->Get(rocksdb::ReadOptions(), "During", &buff);
				CHECK(status.ok()) << status.ToString();
				during = std::stoull(buff);

				// Load Mat Data
				{
					int noc = 0;
					cv::FileStorage fs(folder + "\\data.xml", cv::FileStorage::READ);
					fs["CUMU"] >> cumulative;
					fs["CMat"] >> confusion;
					fs["GTable"] >> global_confusion;
					fs["NOC"] >> noc;
					local_confusions.resize(noc);
					for (int i = 0; i < noc; i++)
					{
						fs["LTable_" + std::to_string(i)] >> local_confusions[i];
					}
				}
			}

			void Run() final
			{
				if (can_run)
				{
					noc = gallery->Size();
					cumulative = CumulativeTabel((int)noc);
					confusion = ConfusionMat((int)noc);
					local_confusions.resize(noc);
					for (auto& c : local_confusions)
					{
						c = ConfusionTable(nbins);
					}

					RunForward(gallery, GALLERY, false);
					RunForward(genuine, GENUINE, true);

					Identify();
				}
				else
				{
					LOG(FATAL) << "Just can show report this test database";
				}
			}
			void Report() final
			{
				// Here must renew mAP
				double mAP = GetMAP(local_confusions, genuine_label);
				double AUC = GetAUC(local_confusions, genuine_label);
				double ACC = confusion.ACC;

				Mat F1 = GetFScore(local_confusions, 1.0, genuine_label);

				Mat FPR = GetFPR(local_confusions, genuine_label);
				Mat TPR = GetTPR(local_confusions, genuine_label);
				Mat PPV = GetPPV(local_confusions, genuine_label);

				auto ths = LineSpace(0, 1, global_confusion.nbins);

				double max_f1_score;
				int max_f1_idx[2];
				cv::minMaxIdx(F1, nullptr, &max_f1_score, nullptr, max_f1_idx);

				int th_base_pr_idx[2];
				cv::Mat diff = cv::abs(TPR - PPV);
				auto ptr = diff.ptr<float>();
				for (int i = 1; i < global_confusion.nbins; i++)
				{
					if (ptr[i] - ptr[i - 1] > 0)
					{
						th_base_pr_idx[0] = 0;
						th_base_pr_idx[1] = i;
						break;
					}
				}

				//std::vector<cv::Vec3f> table;
				std::stringstream table;
				std::vector<float> cond = { 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f };
				for (auto x : cond)
				{
					auto y = Interpolate(FPR, TPR, x);
					auto th = Interpolate(FPR, ths, x);
					table << "  |" << std::scientific << std::setprecision(1) << x
						<< "|" << std::fixed << std::setprecision(4) << y
						<< "@th=" << std::fixed << std::setprecision(2) << th << "|" << std::endl;
				}

				LOG(INFO) << std::endl
					<< "Total Sampels: " << valid_size[0] + valid_size[1] <<  std::endl
					<< "Gallery: " << valid_size[0] << ", Genuine(IDs): " << valid_size[1] << "(" << genuine_label.size() << ")" << std::endl
					<< "Forward Time: " << during / cv::getTickFrequency() / (valid_size[0] + (double)valid_size[1]) * 1000 << " ms/sample" << std::endl
					<< "ACC: " << ACC << std::endl
					<< "AUC: " << AUC << std::endl
					<< "mAP: " << mAP << std::endl
					<< "Optimal Threshold based on PR Curve: " << ths.at<float>(th_base_pr_idx) << std::endl
					<< "Max F1 Score: " << max_f1_score << std::endl
					<< "Optimal Threshold based on F1 Score: " << ths.at<float>(max_f1_idx) << std::endl
					<< "Table of FPR|TPR@TH:" << std::endl << table.str();

				report << "# Test Report" << std::endl
					<< "## Database Infos" << std::endl
					<< " - Total samples: " << valid_size[0] + valid_size[1] << std::endl
					<< " - Gallery: " << valid_size[0] << ", Genuine(IDs): " << valid_size[1] << "(" << genuine_label.size() << ")" << std::endl << std::endl
					<< "## Forward Time" << std::endl
					<< " - Time per forward: " << during / cv::getTickFrequency() / (valid_size[0] + (double)valid_size[1]) * 1000 << " ms" << std::endl << std::endl
					<< "## Evaluation Index" << std::endl
					<< "### FPR|TPR@TH Table" << std::endl
					<< "  |FPR|TPR@TH|" << std::endl
					<< "  |:---:|:---:|" << std::endl
					<< table.str() << std::endl
					<< "### Evaluation" << std::endl
					<< " - ACC: " << ACC << std::endl
					<< " - AUC: " << AUC << std::endl
					<< " - mAP: " << mAP << std::endl
					<< " - Max F1 Score: " << max_f1_score << std::endl
					<< "### Optimal Threshold" << std::endl
					<< " - Optimal Threshold based on PR Curve: " << ths.at<float>(th_base_pr_idx) << std::endl
					<< " - Optimal Threshold based on F1 Score: " << ths.at<float>(max_f1_idx) << std::endl
					<< "### Curves" << std::endl
					<< "#### Cumulative Match Curve" << std::endl
					<< "![CMC Curve](cmc.png)" << std::endl
					<< "#### Receiver Operating Characteristic" << std::endl
					<< "![ROC Curve](roc.png)" << std::endl
					<< "#### PR Cruve" << std::endl
					<< "![PR Curve](prc.png)" << std::endl;
			}

			void Save()
			{
				// Save Mat Data
				{
					cv::FileStorage fs(folder + "\\data.xml", cv::FileStorage::WRITE);
					fs << "CUMU" << cumulative;
					fs << "CMat" << confusion;
					fs << "GTable" << global_confusion;
					fs << "NOC" << (int)noc;
					for (int i = 0; i < noc; i++)
					{
						fs << "LTable_" + std::to_string(i) << local_confusions[i];
					}
				}

				// To save Thumbnail images
				{
					auto cmc = PlotCMC(false)->Draw(); cv::resize(cmc, cmc, cv::Size(), 0.6, 0.6);
					auto roc = PlotROC(false)->Draw(); cv::resize(roc, roc, cv::Size(), 0.6, 0.6);
					auto prc = PlotPRC(false)->Draw(); cv::resize(prc, prc, cv::Size(), 0.6, 0.6);

					cv::imwrite(folder + "\\cmc.png", cmc);
					cv::imwrite(folder + "\\roc.png", roc);
					cv::imwrite(folder + "\\prc.png", prc);
				}

				// Save Reports
				{
					std::fstream fs(folder + "\\report.md", std::ios::out);
					fs << report.str() << std::endl;
					fs << "---" << std::endl;
					fs << "Powered by " << GetVersionInfo();
					fs.close();
				}

				status = database->Put(rocksdb::WriteOptions(), "NOC", std::to_string(noc));
				CHECK(status.ok()) << status.ToString();

				std::vector<int> valid_idx;
				for (auto label : genuine_label) valid_idx.push_back(label);
				status = database->Put(rocksdb::WriteOptions(), "Genuine Label", std::string((char*)valid_idx.data(), sizeof(int) * valid_idx.size()));
				CHECK(status.ok()) << status.ToString();

				status = database->Put(rocksdb::WriteOptions(), "Valid Size", std::string((char*)valid_size, sizeof(int) * 2));
				CHECK(status.ok()) << status.ToString();

				status = database->Put(rocksdb::WriteOptions(), "During", std::to_string(during));
				CHECK(status.ok()) << status.ToString();
			}

			Ptr<PlotFigure> PlotROC(bool show) final
			{
				auto figure = PlotFigure::Figure("ROC");
				figure->Hold(ON);

				Mat TPR, FPR;

				FPR = global_confusion.FPR;
				TPR = global_confusion.TPR;
				Mapping2LogSpace(FPR, TPR);
				figure->Plot(FPR, TPR, "LineWidth", 2, "Legend", "Global");

				FPR = GetFPR(local_confusions, genuine_label);
				TPR = GetTPR(local_confusions, genuine_label);
				Mapping2LogSpace(FPR, TPR);
				figure->Plot(FPR, TPR, "LineWidth", 2, "Legend", "Local");

				figure->Set("XLabel", "FPR(LOG10 SPACE)", "YLabel", "TPR", "Title", "ROC", "YRange", Range(0, 1));
				if (show) figure->Show();

				return figure;
			}
			Ptr<PlotFigure> PlotCMC(bool show) final
			{
				auto figure = PlotFigure::Figure("CMC");
				figure->Hold(ON);

				figure->Plot(cumulative.GetPrecision(), "LineWidth", 2);
				figure->Set("XLabel", "Rank", "YLabel", "Precision", "Title", "CMC", "YRange", Range(0, 1));
				if (show) figure->Show();

				return figure;
			}
			Ptr<PlotFigure> PlotPRC(bool show) final
			{
				auto figure = PlotFigure::Figure("PRC");
				figure->Hold(ON);

				Mat TPR, PPV;

				PPV = global_confusion.PPV;
				TPR = global_confusion.TPR;
				figure->Plot(TPR, PPV, "LineWidth", 2, "Legend", "Global");

				PPV = GetPPV(local_confusions, genuine_label);
				TPR = GetTPR(local_confusions, genuine_label);
				figure->Plot(TPR, PPV, "LineWidth", 2, "Legend", "Local");

				figure->Set("XLabel", "TPR", "YLabel", "PPV", "Title", "PRC", "YRange", Range(0, 1), "XRange", Range(0, 1));
				if (show) figure->Show();

				return figure;
			}

			std::set<int> GetGenuineLabels() const final
			{
				return genuine_label;
			}

			void Close() final
			{
				if (!database) return;

				for (auto& handle : handles)
				{
					delete handle;
					handle = nullptr;
				}
				status = database->Close();
				CHECK(status.ok()) << status.ToString();

				delete database;
				database = nullptr;
			}

		private:
			void RunForward(Ptr<DataLoader>& loader, int idx, bool allow_missing)
			{
				loader->Reset();
				TestData data;
				ProgressBar::Render("Forwarding " + loader->Name(), loader->Size());
				while (!(data = loader->Next()).Empty())
				{
					int64 tick = cv::getTickCount();
					dnn::Tensor feat = forward(data.sample.GetData()[0]);
					during += (cv::getTickCount() - tick);

					if (feat.data)
					{
						CHECK(feat.IsContinue());
						CHECK_EQ(F32, feat.depth);

						status = database->Put(rocksdb::WriteOptions(), handles[idx], data.key, std::string((char*)feat.data, feat.Size() * sizeof(float)));
						CHECK(status.ok()) << status.ToString();
						valid_size[idx]++;
					}
					else
					{
						allow_missing ? 
							(LOG(WARNING) << "Can not fetch feature from " << data.key << "th sample in " << loader->Name()) : 
							(LOG(FATAL) << loader->Name() << " do not allow missing, " << data.key);
					}
					ProgressBar::Update();
				}
				ProgressBar::Halt();
			}

			void Identify()
			{
				std::vector<rocksdb::Iterator*> iters; // 0->gallery, 1->genuine
				status = database->NewIterators(rocksdb::ReadOptions(), handles, &iters);
				CHECK(status.ok()) << status.ToString();

				auto Match = [=](const Mat& f1) {
					std::vector<double> scores(noc);
					for (iters[GALLERY]->SeekToFirst(); iters[GALLERY]->Valid(); iters[GALLERY]->Next())
					{
						auto key = iters[GALLERY]->key().ToString();
						int id = gallery->Get(key).label.CastTo<CLabel>()[0];
						auto buff = iters[GALLERY]->value().ToString();
						int len = (int)buff.size() / sizeof(float);
						cv::Mat f2(1, len, CV_32F, buff.data());
						scores[id] = measure(f1, f2);
					}
					return scores;
				}; // Slow ?   ---->  Yes, slow!!

				ProgressBar::Render("Identifying", valid_size[GENUINE]);
				for (iters[GENUINE]->SeekToFirst(); iters[GENUINE]->Valid(); iters[GENUINE]->Next())
				{
					auto key = iters[GENUINE]->key().ToString();

					auto buffer = iters[GENUINE]->value().ToString();
					int len = (int)buffer.size() / sizeof(float);
					cv::Mat feat(1, len, CV_32F, buffer.data());

					std::vector<double> scores = Match(feat);

					int predict_id = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
					int actual_id = (int)genuine->Get(key).label.CastTo<CLabel>()[0];

					genuine_label.insert(actual_id);

					cumulative.Apply(actual_id, scores);
					confusion.Apply(actual_id, predict_id);
					for (int i = 0; i < scores.size(); i++)
					{
						local_confusions[i].Apply(actual_id == i, scores[i]);
						global_confusion.Apply(actual_id == i, scores[i]); // Micro Averaging
					}
					ProgressBar::Update();
				}
				ProgressBar::Halt();				

				for (auto& iter : iters)
				{
					delete iter;
					iter = nullptr;
				}
			}

			

			void Mapping2LogSpace(Mat& FPR, Mat& TPR)
			{
				auto tpr = TPR.ptr<float>();
				auto fpr = FPR.ptr<float>();

				float x = 1e-4f;
				float y;
				// tpr and fpr is monotone decreasing
				for (int i = 0; i < nbins; i++)
				{
					if (fpr[i] < x)
					{
						float k = (tpr[i - 1] - tpr[i]) / (fpr[i - 1] - fpr[i]);
						float delta_y = k * (fpr[i - 1] - x);
						y = tpr[i - 1] - delta_y;
						break;
					}
				}
				// truncate
				for (int i = 0; i < nbins; i++)
				{
					if (fpr[i] < x)
					{
						fpr[i] = x;
						tpr[i] = y;
					}
				}

				//x -> Log Space
				for (int i = 0; i < nbins; i++)
				{
					fpr[i] = log10(fpr[i] * 100);
				}
			}

			rocksdb::DB* database; // For features
			rocksdb::Status status;
			std::vector<rocksdb::ColumnFamilyHandle*> handles;// 0->gallery, 1->genuine

			int nbins = 1000;
			size_t noc = 0;
			int valid_size[2] = {0, 0};
			std::set<int> genuine_label;
			int64 during = 0;
			std::stringstream report;
			bool can_run = false;
		};


		Ptr<ITest> ITest::Create(const std::string& db)
		{
			auto engine = std::make_shared<Identification>();
			engine->Create(db);
			return engine;
		}

		Ptr<ITest> ITest::Load(const std::string& db)
		{
			auto engine = std::make_shared<Identification>();
			engine->Load(db);
			return engine;
		}
	}
}