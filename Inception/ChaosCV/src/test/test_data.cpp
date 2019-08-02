#include "test/test_data.hpp"

#include <rocksdb/db.h>

namespace chaos
{
	namespace test
	{

		CLabel::LabelData::LabelData() {}
		CLabel::LabelData& CLabel::LabelData::operator,(const int& value)
		{
			values.push_back(value);
			return *this;
		}
		CLabel::CLabel() : data(LabelData()) {}
		CLabel::CLabel(const CLabel::LabelData& data) : data(data) {}
		CLabel::CLabel(const std::string& buff)
		{
			//size_t cnt = *(size_t*)buff.data();
			size_t cnt = buff.size() / sizeof(int);
			auto values = (int*)buff.data();
			for (size_t i = 0; i < cnt; i++)
			{
				data.values.push_back(values[i]);
			}
		}
		CLabel::LabelData& CLabel::operator<<(const int& value) { return data, value; }
		int CLabel::operator[](int i) const
		{
			return data.values[i];
		}


		DLabel::Value::Value(int idx, const Rect& rect) : idx(idx), rect(rect) {}
		DLabel::LabelData::LabelData() {}
		DLabel::LabelData& DLabel::LabelData::operator,(const DLabel::Value& value)
		{
			values.push_back(value);
			return *this;
		}

		DLabel::DLabel() {}
		DLabel::DLabel(const DLabel::LabelData& data) : data(data) {}
		DLabel::DLabel(const std::string& buff)
		{
			//size_t cnt = *(size_t*)buff.data();
			size_t cnt = buff.size() / sizeof(Value);
			auto values = (Value*)buff.data();
			for (size_t i = 0; i < cnt; i++)
			{
				data.values.push_back(values[i]);
			}
		}
		DLabel::LabelData& DLabel::operator<<(const DLabel::Value& value)
		{
			return data, value;
		}
		DLabel::Value DLabel::operator[](int i) const
		{
			return data.values[i];
		}


		Label::Label() : type(CLABEL) {}
		//Label::Label(const Label& label) : data(label.data), type(label.type) {}

		Label::Label(const CLabel& label) : type(CLABEL)
		{
			size_t size = label.data.values.size();
			data += std::string((char*)label.data.values.data(), size * sizeof(int));
		}
		Label::Label(const CLabel::LabelData& label_data) : type(CLABEL)
		{
			size_t size = label_data.values.size();
			data += std::string((char*)label_data.values.data(), size * sizeof(int));
		}

		Label::Label(const DLabel& label) : type(DLABEL)
		{
			size_t size = label.data.values.size();
			data += std::string((char*)label.data.values.data(), size * sizeof(DLabel::Value));
		}
		Label::Label(const DLabel::LabelData& label_data) : type(DLABEL)
		{
			size_t size = label_data.values.size();
			data += std::string((char*)label_data.values.data(), size * sizeof(DLabel::Value));
		}

		Label::Label(const std::string& buff)
		{
			type = *(Type*)buff.data();
			data = buff.substr(sizeof(Type));
		}
		std::string Label::ToString() const
		{
			std::string buff;
			buff += std::string((char*)& type, sizeof(Type));
			buff += data;
			return buff;
		}
		bool Label::Empty() const
		{
			return data.empty();
		}




		Sample::Sample() : type(FILE) {}
		Sample::Sample(const FileList& group) : data(std::string()), type(FILE)
		{
			size_t cnt = group.size();
			data += std::string((char*)&cnt, sizeof(size_t));

			for (auto file : group)
			{
				std::string f = file;
				size_t size = f.size();

				data += std::string((char*)&size, sizeof(size_t));
				data += f;
			}
		}
		Sample::Sample(const std::vector<Mat>& group) : data(std::string()), type(DATA) 
		{
			size_t cnt = group.size();
			data += std::string((char*)&cnt, sizeof(size_t));

			for (auto value : group)
			{
				int depth = value.depth();
				int channel = value.channels();
				int dims = value.dims;

				data += std::string((char*)&depth, sizeof(int));
				data += std::string((char*)&channel, sizeof(int));
				data += std::string((char*)&dims, sizeof(int));

				for (int i = 0; i < dims; i++)
				{
					int shape = value.size[i];
					data += std::string((char*)&shape, sizeof(int));
				}

				data += std::string((char*)value.data, value.total() * value.elemSize());
			}
		}

		Sample::Sample(const std::string& buff)
		{
			type = *(Type*)buff.data();
			data = buff.substr(sizeof(Type));
		}
		std::string Sample::ToString() const
		{
			std::string buff;
			buff += std::string((char*)&type, sizeof(Type));
			buff += data;
			return buff;
		}
		std::vector<Mat> Sample::GetData() const
		{
			std::vector<Mat> group;
			
			if (type == DATA)
			{
				const char* values = data.data();
				size_t cnt = *(size_t*)values; values += sizeof(size_t);
				for (int i = 0; i < cnt; i++)
				{
					int depth = *(int*)values; values += sizeof(int);
					int channel = *(int*)values; values += sizeof(int);
					int dims = *(int*)values; values += sizeof(int);
					std::vector<int> shape;
					for (int i = 0; i < dims; i++)
					{
						shape.push_back(*(int*)values); values += sizeof(int);
					}

					Mat value = Mat(dims, shape.data(), CV_MAKETYPE(depth, channel), (void*)values);
					values += value.total() * value.elemSize();

					group.push_back(value);
				}
			}
			else // type == FILE
			{
				const char* values = data.data();
				size_t cnt = *(size_t*)values; values += sizeof(size_t);
				for (int i = 0; i < cnt; i++)
				{
					size_t size = *(size_t*)values; values += sizeof(size_t);
					std::string file = std::string(values, size); values += size;

					cv::Mat value = cv::imread(file);
					group.push_back(value);
				}
			}

			return group;
		}
		bool Sample::Empty() const
		{
			return data.empty();
		}

		


		TestData::TestData() {}
		TestData::TestData(const std::string& key, const Sample& sample, const Label& label) : key(key), sample(sample), label(label) {}
		bool TestData::Empty() const
		{
			return sample.Empty() || label.Empty();
		}


		class DBWriter : public DataWriter
		{
		public:
			DBWriter(const std::string& db)
			{
				rocksdb::Options options;
				options.create_if_missing = true;
				options.error_if_exists = true;

				std::vector<rocksdb::ColumnFamilyDescriptor> descriptors = {
					rocksdb::ColumnFamilyDescriptor("Sample", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor("Label", rocksdb::ColumnFamilyOptions())
				};

				status = rocksdb::DB::Open(options, db, &database);
				CHECK(status.ok()) << status.ToString();
				status = database->CreateColumnFamilies(descriptors, &handles);
				CHECK(status.ok()) << status.ToString();
			}

			~DBWriter()
			{
				Close();
			}

			void Put(const Sample& sample, const Label& label) final
			{
				std::string key = std::to_string(size);
				status = database->Put(rocksdb::WriteOptions(), handles[0], key, sample.ToString());
				CHECK(status.ok()) << status.ToString();
				status = database->Put(rocksdb::WriteOptions(), handles[1], key, label.ToString());
				CHECK(status.ok()) << status.ToString();

				size++;
			}

			void Close() final
			{
				if (!database) return;

				status = database->Put(rocksdb::WriteOptions(), "Size", std::to_string(size));
				CHECK(status.ok()) << status.ToString();

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

			size_t Size() const final
			{
				return size;
			}

			std::string Name() const
			{
				return database->GetName();
			}

		private:
			rocksdb::DB* database;
			rocksdb::Status status;
			std::vector<rocksdb::ColumnFamilyHandle*> handles;

			size_t size = 0;
		};
		Ptr<DataWriter> DataWriter::Create(const std::string& db)
		{
			return Ptr<DataWriter>(new DBWriter(db));
		}


		class DBLoader : public DataLoader
		{
		public:
			DBLoader(const std::string& db)
			{
				std::vector<rocksdb::ColumnFamilyDescriptor> descriptors = {
					rocksdb::ColumnFamilyDescriptor("Sample", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor("Label", rocksdb::ColumnFamilyOptions()),
					rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName, rocksdb::ColumnFamilyOptions())
				};

				status = rocksdb::DB::Open(rocksdb::Options(), db, descriptors, &handles, &database);
				CHECK(status.ok()) << status.ToString();
				//handles.pop_back(); // Remove default column

				status = database->NewIterators(rocksdb::ReadOptions(), handles, &iters);
				CHECK(status.ok()) << status.ToString();
				for (auto iter : iters)
				{
					iter->SeekToFirst();
				}

				std::string value;
				status = database->Get(rocksdb::ReadOptions(), "Size", &value);
				CHECK(status.ok()) << status.ToString();

				size = std::stoull(value);//*(size_t*)value.data();
			}

			TestData Get(const std::string& key) final
			{
				CHECK_LT(std::stoull(key), size) << "Key is out of range";

				Sample sample;
				Label label;

				std::string sa_value, la_value;
				status = database->Get(rocksdb::ReadOptions(), handles[0], key, &sa_value);
				CHECK(status.ok()) << status.ToString();
				sample = sa_value;

				status = database->Get(rocksdb::ReadOptions(), handles[1], key, &la_value);
				CHECK(status.ok()) << status.ToString();
				label = la_value;

				return TestData(key, sample, label);
			}

			TestData Next() final
			{
				TestData data;
				if (iters[0]->Valid() && iters[1]->Valid())
				{
					CHECK_EQ(iters[0]->key(), iters[1]->key());
					std::string key = iters[0]->key().ToString();
					Sample sample = iters[0]->value().ToString();
					Label label = iters[1]->value().ToString();

					data = TestData(key, sample, label);

					iters[0]->Next();
					iters[1]->Next();
				}
				return data;
			}

			void Reset() final
			{
				for (auto iter : iters)
				{
					iter->Reset();
					iter->SeekToFirst();
				}
			}

			~DBLoader()
			{
				Close();
			}

			void Close() final
			{
				if (!database) return;

				for (auto& handle : handles)
				{
					delete handle;
					handle = nullptr;
				}
				for (auto& iter : iters)
				{
					delete iter;
					iter = nullptr;
				}

				status = database->Close();
				CHECK(status.ok()) << status.ToString();

				delete database;
				database = nullptr;
			}

			size_t Size() const final
			{
				return size;
			}

			std::string Name() const
			{
				return database->GetName();
			}

		private:
			rocksdb::DB* database;
			rocksdb::Status status;

			// 0->sample; 1->label;
			std::vector<rocksdb::ColumnFamilyHandle*> handles;
			// 0->sample; 1->label;
			std::vector<rocksdb::Iterator*> iters;

			size_t size = 0;
		};
		Ptr<DataLoader> DataLoader::Load(const std::string& db)
		{
			return Ptr<DataLoader>(new DBLoader(db));
		}
	}
}