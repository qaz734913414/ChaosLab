#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	namespace test
	{
		/// <summary>Classification Label</summary>
		class CHAOS_API CLabel
		{
		public:
			class CHAOS_API LabelData
			{
			public:
				LabelData();
				LabelData& operator,(const int& value);

				std::vector<int> values;
			};

			CLabel();
			CLabel(const LabelData& data);
			CLabel(const std::string& buff);

			LabelData& operator<<(const int& value);
			int operator[](int i) const;

			LabelData data;
		};

		/// <summary>Detection Label</summary>
		class CHAOS_API DLabel
		{
		public:
			class CHAOS_API Value
			{
			public:
				Value(int idx, const Rect& rect);

				int idx;
				Rect rect;
			};

			class CHAOS_API LabelData
			{
			public:
				LabelData();
				LabelData& operator,(const Value& value);

				std::vector<Value> values;
			};

			DLabel();
			DLabel(const LabelData& data);
			DLabel(const std::string& buff);
			LabelData& operator<<(const Value& value);
			Value operator[](int i) const;

			LabelData data;
		};

		// The precedence of , is lower than <<, so must use (int,Rect) to get DLabel::LabelData first
		static inline DLabel::Value operator,(int idx, const Rect& rect)
		{
			return DLabel::Value(idx, rect);
		}

		/// <summary>Test Label</summary>
		class CHAOS_API Label
		{
		public:
			enum Type
			{
				CLABEL,
				DLABEL,
			};

			Label();

			Label(const CLabel& label);
			Label(const CLabel::LabelData& label_data);

			Label(const DLabel& label);
			Label(const DLabel::LabelData& label_data);

			Label(const std::string& buff);
			//operator std::string() const;

			std::string ToString() const;

			template<class _Tp>
			_Tp CastTo()
			{
				switch (type)
				{
				case CLABEL:
					CHECK_EQ(typeid(_Tp), typeid(CLabel)) << "Can not cast CLabel to " << typeid(_Tp).name();
					break;
				case DLABEL:
					CHECK_EQ(typeid(_Tp), typeid(DLabel)) << "Can not cast DLabel to " << typeid(_Tp).name();
					break;
				default:
					LOG(FATAL);
					return _Tp(); // Never reachable
				}

				return data;
			}

			bool Empty() const;
		private:
			std::string data;
			Type type;
		};


		/// <summary>Test Sample</summary>
		class CHAOS_API Sample
		{
		public:
			enum Type
			{
				FILE, /// Image file
				DATA, /// Mat data
			};

			Sample();
			Sample(const FileList& group);
			Sample(const std::vector<Mat>& group);

			Sample(const std::string& buff);
			//operator std::string() const;

			std::string ToString() const;

			std::vector<Mat> GetData() const;

			bool Empty() const;
		private:
			std::string data;
			Type type;
		};
		


		class CHAOS_API TestData
		{
		public:
			TestData();
			TestData(const std::string& key, const Sample& sample, const Label& label);
			bool Empty() const;

			std::string key;
			Sample sample;
			Label label;
		};



		class CHAOS_API Database
		{
		public:
			virtual void Close() = 0;
			virtual size_t Size() const = 0;
			virtual std::string Name() const = 0;
		};

		class CHAOS_API DataWriter : public Database
		{
		public:
			virtual ~DataWriter() {}

			virtual void Put(const Sample& sample, const Label& label) = 0;

			static Ptr<DataWriter> Create(const std::string& db);
		};

		class CHAOS_API DataLoader : public Database
		{
		public:
			virtual ~DataLoader() {}
			
			virtual TestData Get(const std::string& key) = 0;
			virtual TestData Next() = 0;
			virtual void Reset() = 0;

			static Ptr<DataLoader> Load(const std::string& db);
		};

		
	}
}