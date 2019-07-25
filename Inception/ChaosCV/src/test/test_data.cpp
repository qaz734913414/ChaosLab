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
		CLabel::LabelData& CLabel::operator<<(const int& value) { return data, value; }



		DLabel::Value::Value(int idx, const Rect& rect) : idx(idx), rect(rect) {}
		DLabel::LabelData::LabelData() {}
		DLabel::LabelData& DLabel::LabelData::operator,(const DLabel::Value& value)
		{
			values.push_back(value);
			return *this;
		}

		DLabel::DLabel() {}
		DLabel::DLabel(const DLabel::LabelData& data) : data(data) {}
		DLabel::LabelData& DLabel::operator<<(const DLabel::Value& value)
		{
			return data, value;
		}






		Label::Label() : type(CLABEL) {}
		//Label::Label(const Label& label) : data(label.data), type(label.type) {}

		Label::Label(const CLabel& label) : type(CLABEL)
		{
			size_t size = label.data.values.size();
			data = std::string((char*)label.data.values.data(), size * sizeof(int));
		}
		Label::Label(const CLabel::LabelData& label_data) : type(CLABEL)
		{
			size_t size = label_data.values.size();
			data = std::string((char*)label_data.values.data(), size * sizeof(int));
		}

		Label::Label(const DLabel& label) : type(DLABEL)
		{
			size_t size = label.data.values.size();
			data = std::string((char*)label.data.values.data(), size * sizeof(DLabel::Value));
		}
		Label::Label(const DLabel::LabelData& label_data) : type(DLABEL)
		{
			size_t size = label_data.values.size();
			data = std::string((char*)label_data.values.data(), size * sizeof(DLabel::Value));
		}

		Label::Label(const std::string& buff)
		{
			type = *(Type*)buff.data();
			data = buff.substr(sizeof(Type));
		}
		
		Label::operator std::string() const
		{
			std::string buff;
			buff += std::string((char*)&type, sizeof(Type));
			buff += data;
			return buff;
		}




		Sample::Sample() : data(Mat()), type(FILE) {}
		Sample::Sample(const File& file) : type(FILE)
		{
			std::string buff = file;
			data = Mat(1, (int)buff.size(), CV_8U, buff.data()).clone();
			//memcpy(data.data, buff.data(), buff.size());
		} 
		Sample::Sample(const Mat& data) : data(data), type(DATA) {}


		Sample::operator std::string() const 
		{
			return std::string((char*)data.data, data.total() * data.channels());
		}
		bool Sample::IsData() const
		{
			return type == DATA;
		}

		


		TestData::TestData() {}
		TestData::TestData(const Label& label, const Sample& sample) : label(label), sample(sample)
		{
			
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

			size_t Size() final
			{
				return size;
			}

		private:
			rocksdb::DB* database;
			rocksdb::Status status;
			std::vector<rocksdb::ColumnFamilyHandle*> handles;

			size_t size;
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

				size = *(size_t*)value.data();
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

			size_t Size() final
			{
				return size;
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