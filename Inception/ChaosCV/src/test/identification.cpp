#include "test/test_engine.hpp"

#include <rocksdb/db.h>

namespace chaos
{
	namespace test
	{
		ITest::~ITest() {}

		void ITest::SetGallery(const Ptr<DataLoader>& loader) { gallery = loader; }
		void ITest::SetGenuine(const Ptr<DataLoader>& loader) { genuine = loader; }

		void ITest::SetMeasure(const std::function<double(const Mat&, const Mat&)>& func) { measure = func; }


		class Identification : public ITest
		{
		public:
			enum
			{
				GALLERY = 0,
				GENUINE = 1,
			};

			Identification() : database(nullptr) {}
			~Identification()
			{
				Close();
			}


			void Load(const std::string& db)
			{

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
			}

			void Run() final
			{
				noc = gallery->Size();
				confusion = ConfusionMat((int)noc);
				local_confusions.resize(noc);

				RunForward(gallery, GALLERY, false);
				RunForward(genuine, GENUINE, true);

				Identify();
			}
			void Report() final
			{
				double mAP = GetMAP(local_confusions);
				status = database->Put(rocksdb::WriteOptions(), "mAP", std::to_string(mAP));
				CHECK(status.ok()) << status.ToString();

				double AUC = global_confusion.AUC;
				status = database->Put(rocksdb::WriteOptions(), "AUC", std::to_string(AUC));
				CHECK(status.ok()) << status.ToString();

				double ACC = confusion.GetACC();
				status = database->Put(rocksdb::WriteOptions(), "ACC", std::to_string(ACC));
				CHECK(status.ok()) << status.ToString();

				//global_confusion.GetFScore();
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

			void RunForward(Ptr<DataLoader>& loader, int idx, bool allow_missing)
			{
				loader->Reset();
				TestData data;
				ProgressBar::Render("Forwarding " + loader->Name(), loader->Size());
				while (!(data = loader->Next()).Empty())
				{
					Mat feat = forward(data.sample.GetData()[0]);
					if (!feat.empty())
					{
						CHECK_EQ(CV_32F, feat.depth());
						CHECK_EQ(4, feat.dims);
						int len = feat.size[0] * feat.size[1] * feat.size[2] * feat.size[3];

						status = database->Put(rocksdb::WriteOptions(), handles[idx], data.key, std::string((char*)feat.data, len * sizeof(float)));
						CHECK(status.ok()) << status.ToString();
						valid_size[idx]++;
					}
					else
					{
						allow_missing ? 
							(LOG(WARNING) << "Can not fetch feature from " << data.key << "th sample in " << loader->Name()) : 
							(LOG(FATAL) << loader->Name() << " do not allow missing");
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
					for (iters[0]->SeekToFirst(); iters[0]->Valid(); iters[0]->Next())
					{
						auto key = iters[0]->key().ToString();
						int id = gallery->Get(key).label.CastTo<CLabel>()[0];
						auto buff = iters[0]->value().ToString();
						int len = (int)buff.size() / sizeof(float);
						cv::Mat f2(1, len, CV_32F, buff.data());
						scores[id] = measure(f1, f2);
					}
					return scores;
				}; // Slow ?

				ProgressBar::Render("Identify", valid_size[GENUINE]);
				for (iters[1]->SeekToFirst(); iters[1]->Valid(); iters[1]->Next())
				{
					auto key = iters[1]->key().ToString();

					auto buffer = iters[1]->value().ToString();
					int len = (int)buffer.size() / sizeof(float);
					cv::Mat feat(1, len, CV_32F, buffer.data());

					std::vector<double> scores = Match(feat);

					int predict_id = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
					int actual_id = (int)genuine->Get(key).label.CastTo<CLabel>()[0];

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

			rocksdb::DB* database; // For features
			rocksdb::Status status;
			std::vector<rocksdb::ColumnFamilyHandle*> handles;// 0->gallery, 1->genuine

			size_t noc = 0;
			int valid_size[2] = {0, 0};
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