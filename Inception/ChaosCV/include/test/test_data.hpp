#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	namespace test
	{
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
			LabelData& operator<<(const int& value);

			LabelData data;
		};

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
			LabelData& operator<<(const Value& value);

			LabelData data;
		};

		// The precedence of , is lower than <<, so must use (int,Rect) to get DLabel::LabelData first
		static inline DLabel::Value operator,(int idx, const Rect& rect)
		{
			return DLabel::Value(idx, rect);
		}


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
			operator std::string() const;
		public:
			std::string data;
			Type type;
		};

		class CHAOS_API Sample
		{
		public:
			enum Type
			{
				FILE,
				DATA,
			};

			Sample();
			Sample(const File& file);
			Sample(const Mat& data);

			operator std::string() const;
			bool IsData() const;
		private:
			Mat data;
			Type type;
		};



		class CHAOS_API TestData
		{
		public:
			TestData();
			TestData(const Label& label, const Sample& sample);

		private:
			Sample sample;
			Label label;
		};





		class CHAOS_API Database
		{
		public:
			virtual void Close() = 0;
			virtual size_t Size() = 0;
		};

		class CHAOS_API DataWriter : public Database
		{
		public:
			virtual ~DataWriter() {}

			
			//virtual void Put(const Sample& sample, const Label& label) = 0;
			static Ptr<DataWriter> Create(const std::string& db);
		};

		class CHAOS_API DataLoader : public Database
		{
		public:
			virtual ~DataLoader() {}
			
			//virtual TestData Next() = 0;

			static Ptr<DataLoader> Load(const std::string& db);
		};

		
	}
}