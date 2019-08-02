#pragma once

#include "test/test_data.hpp"
#include "test/confusion.hpp"
#include "test/cumulative.hpp"

#include "highgui/highgui.hpp"
#include "highgui/plot.hpp"

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

			std::string folder;
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

			

			CumulativeTabel GetCumulative() const;
			ConfusionMat GetConfusion() const;
			ConfusionTable GetGlobalConfusion() const;
			std::vector<ConfusionTable> GetLocalConfusions() const;

			__declspec(property(get = GetCumulative)) CumulativeTabel Cumulative;
			__declspec(property(get = GetConfusion)) ConfusionMat Confusion;
			__declspec(property(get = GetGlobalConfusion)) ConfusionTable GlobalConfusion;
			__declspec(property(get = GetLocalConfusions)) std::vector<ConfusionTable> LocalConfusions;

			virtual Ptr<PlotFigure> PlotROC(bool show = true) = 0;
			virtual Ptr<PlotFigure> PlotCMC(bool show = true) = 0;
			virtual Ptr<PlotFigure> PlotPRC(bool show = true) = 0;

			virtual std::set<int> GetGenuineLabels() const = 0;

			__declspec(property(get = GetGenuineLabels)) std::set<int> GenuineLabels;

			static Ptr<ITest> Create(const std::string& db);
			static Ptr<ITest> Load(const std::string& db);
		protected:
			enum
			{
				GALLERY = 0,
				GENUINE = 1,
			};

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