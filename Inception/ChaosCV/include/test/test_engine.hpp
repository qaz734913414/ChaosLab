#pragma once

#include "test/test_data.hpp"
#include "test/confusion.hpp"
#include "test/cumulative.hpp"

namespace chaos
{
	namespace test
	{
		class CHAOS_API TestEngine
		{
		public:
			virtual void Run() = 0;
			virtual void Report() = 0;
			virtual void Close() = 0;

			void SetForward(const std::function<Mat(const Mat&)>& func) { forward = func; }
			__declspec(property(put = SetForward)) std::function<Mat(const Mat&)> Forward;
		protected:
			std::function<Mat(const Mat&)> forward;
		};

		/// <summary>
		/// <para>Identification Test</para>
		/// <para></para>
		/// </summary>
		class CHAOS_API ITest : public TestEngine
		{
		public:
			virtual ~ITest();

			void SetGallery(const Ptr<DataLoader>& loader);
			void SetGenuine(const Ptr<DataLoader>& loader);

			void SetMeasure(const std::function<double(const Mat&, const Mat&)>& func);

			__declspec(property(put = SetGallery)) Ptr<DataLoader> Gallery;
			__declspec(property(put = SetGenuine)) Ptr<DataLoader> Genuine;
			__declspec(property(put = SetMeasure)) std::function<double(const Mat&, const Mat&)> Measure;

			static Ptr<ITest> Create(const std::string& db);
			static Ptr<ITest> Load(const std::string& db);
		protected:
			Ptr<DataLoader> gallery;
			Ptr<DataLoader> genuine;

			// Default measure method is COS distance
			std::function<double(const Mat&, const Mat&)> measure = [](const Mat& f1, const Mat& f2) {
				auto dis = f1.dot(f2) / sqrt(f1.dot(f1)) / sqrt(f2.dot(f2));
				return (dis + 1.) / 2.;
			};

			CumulativeTabel cumulative;
			ConfusionMat confusion;
			ConfusionTable global_confusion; // Micro average confusion table
			std::vector<ConfusionTable> local_confusions;
		};
	}
}