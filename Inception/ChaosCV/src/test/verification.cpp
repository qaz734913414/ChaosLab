#include "test/test_engine.hpp"

#include <rocksdb/db.h>

#include <fstream>

namespace chaos
{
	namespace test
	{
		void VTest::SetPairList(const Ptr<DataLoader>& loader) { pair_list = loader; }
		ConfusionTable VTest::GetConfusion() const { return confusion; }

		class Verification : public VTest
		{
		public:
			Verification() : database(nullptr) { confusion = ConfusionTable(nbins); }

			~Verification() { Close();  }

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


			void Create(const std::string& db)
			{
				folder = db;

				rocksdb::Options options;
				options.create_if_missing = true;
				options.error_if_exists = true;

				std::vector<rocksdb::ColumnFamilyDescriptor> descriptors = {
					rocksdb::ColumnFamilyDescriptor("Feat0", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor("Feat1", rocksdb::ColumnFamilyOptions()),
				};

				status = rocksdb::DB::Open(options, db, &database);
				CHECK(status.ok()) << status.ToString();
				status = database->CreateColumnFamilies(descriptors, &handles);
				CHECK(status.ok()) << status.ToString();
			}

			void Load(const std::string& db)
			{
				folder = db;
			}

			void Run() final
			{
				RunForward();
				Verify();
			}

			void Report() final
			{
				// Test
				auto ACC = confusion.ACC;
				auto TPR = confusion.TPR;
				auto FPR = confusion.FPR;
				auto PPV = confusion.PPV;

				auto AUC = confusion.AUC;
				auto AP = confusion.AP;

				auto ths = LineSpace(0, 1, confusion.nbins);

				int max_acc_idx[2];
				double max_acc;
				cv::minMaxIdx(ACC, nullptr, &max_acc, nullptr, max_acc_idx);

				int th_base_pr_idx[2];
				cv::Mat diff = cv::abs(TPR - PPV);
				auto ptr = diff.ptr<float>();
				for (int i = 1; i < confusion.nbins; i++)
				{
					if (ptr[i] - ptr[i - 1] > 0)
					{
						th_base_pr_idx[0] = 0;
						th_base_pr_idx[1] = i;
						break;
					}
				}
				
				LOG(INFO) << std::endl
					<< "Max ACC: " << max_acc << std::endl
					<< "Threshold based on ACC: " << ths.at<float>(max_acc_idx) << std::endl
					<< "AUC: " << AUC << std::endl
					<< "AP: " << AP << std::endl
					<< "Threshold based on PR: " << ths.at<float>(th_base_pr_idx);
			}

			void Save() final
			{
				// Test

			}

		private:
			void RunForward()
			{
				pair_list->Reset();
				TestData data;
				ProgressBar::Render("Forwarding " + pair_list->Name(), pair_list->Size());
				while (!(data = pair_list->Next()).Empty())
				{
					auto group = data.sample.GetData();

					dnn::Tensor feat0 = forward(group[0]);
					//std::cout << feat0 << std::endl;

					dnn::Tensor feat1 = forward(group[1]);

					if (feat0.data && feat1.data)
					{
						CHECK(feat0.IsContinue());
						CHECK(feat1.IsContinue());
						CHECK_EQ(F32, feat0.depth);
						CHECK_EQ(F32, feat1.depth);

						status = database->Put(rocksdb::WriteOptions(), handles[0], data.key, std::string((char*)feat0.data, feat0.Size() * sizeof(float)));
						CHECK(status.ok()) << status.ToString();
						status = database->Put(rocksdb::WriteOptions(), handles[1], data.key, std::string((char*)feat1.data, feat1.Size() * sizeof(float)));
						CHECK(status.ok()) << status.ToString();
					}
					else
					{
						LOG(FATAL) << pair_list->Name() << " do not allow missing, " << data.key;
					}
					ProgressBar::Update();
				}
				ProgressBar::Halt();
			}

			void Verify()
			{
				std::vector<rocksdb::Iterator*> iters;
				status = database->NewIterators(rocksdb::ReadOptions(), handles, &iters);
				CHECK(status.ok()) << status.ToString();

				for (iters[0]->SeekToFirst(), iters[1]->SeekToFirst(); 
					 iters[0]->Valid() && iters[1]->Valid(); 
					 iters[0]->Next(), iters[1]->Next())
				{
					CHECK_EQ(iters[0]->key().ToString(), iters[1]->key().ToString());
					std::string key = iters[0]->key().ToString();

					auto buffer0 = iters[0]->value().ToString();
					int len0 = (int)buffer0.size() / sizeof(float);
					cv::Mat feat0(1, len0, CV_32F, buffer0.data());

					auto buffer1 = iters[1]->value().ToString();
					int len1 = (int)buffer1.size() / sizeof(float);
					cv::Mat feat1(1, len1, CV_32F, buffer1.data());

					auto score = measure(feat0, feat1);

					int label = pair_list->Get(key).label.CastTo<CLabel>()[0];

					confusion.Apply(label, score);
				}

				for (auto& iter : iters)
				{
					delete iter;
					iter = nullptr;
				}
			}

			rocksdb::DB* database; // For features
			rocksdb::Status status;
			std::vector<rocksdb::ColumnFamilyHandle*> handles;// 0->feat0, 1->feat1

			int nbins = 1000;
		};

		

		Ptr<VTest> VTest::Create(const std::string& db)
		{
			auto engine = std::make_shared<Verification>();
			engine->Create(db);
			return engine;
		}
		Ptr<VTest> VTest::Load(const std::string& db)
		{
			auto engine = std::make_shared<Verification>();
			engine->Load(db);
			return engine;
		}
	}
}