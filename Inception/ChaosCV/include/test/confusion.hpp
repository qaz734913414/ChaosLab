#pragma once

#include "core/core.hpp"

namespace chaos
{
	namespace test
	{
		class CHAOS_API ConfusionTable
		{
		public:
			ConfusionTable(int nbins = 1000);
			ConfusionTable(const Mat& table);

			void Apply(bool is_positive, double prob);

			CHAOS_API friend void operator>>(const cv::FileNode& node, ConfusionTable& table)
			{
				Mat data;
				cv::read(node, data, Mat());
				table = data;
			}

			CHAOS_API friend cv::FileStorage& operator<<(cv::FileStorage& fs, const ConfusionTable& c)
			{
				if (!fs.isOpened())
					return fs;
				if (fs.state == cv::FileStorage::NAME_EXPECTED + cv::FileStorage::INSIDE_MAP)
					LOG(FATAL) << "No element name has been given";
				write(fs, fs.elname, c.table);
				if (fs.state & cv::FileStorage::INSIDE_MAP)
					fs.state = cv::FileStorage::NAME_EXPECTED + cv::FileStorage::INSIDE_MAP;
				return fs;
			}

			Mat GetTPR() const;
			Mat GetFPR() const;
			Mat GetPPV() const;
			Mat GetNPV() const;

			Mat GetACC() const;

			double GetAP() const;
			double GetAUC() const;

			Mat GetFScore(double beta = 1) const;

			__declspec(property(get = GetTPR)) Mat TPR;
			__declspec(property(get = GetFPR)) Mat FPR;
			__declspec(property(get = GetPPV)) Mat PPV;
			__declspec(property(get = GetNPV)) Mat NPV;
			__declspec(property(get = GetACC)) Mat ACC;

			__declspec(property(get = GetAP)) double AP;
			__declspec(property(get = GetAUC)) double AUC;

			Mat table; // 4 X NBINS
			int nbins = 1000;
		};

		class CHAOS_API ConfusionMat
		{
		public:
			ConfusionMat();
			ConfusionMat(int noc);
			ConfusionMat(const cv::SparseMat& cmat);

			void Apply(int actual_id, int predict_id);

			double GetACC() const;

			__declspec(property(get = GetACC)) double ACC;

			CHAOS_API friend void operator>>(const cv::FileNode& node, ConfusionMat& confusion)
			{
				cv::SparseMat data;
				cv::read(node, data, cv::SparseMat());
				confusion = data;
			}

			CHAOS_API friend cv::FileStorage& operator<<(cv::FileStorage& fs, const ConfusionMat& c)
			{
				if (!fs.isOpened())
					return fs;
				if (fs.state == cv::FileStorage::NAME_EXPECTED + cv::FileStorage::INSIDE_MAP)
					LOG(FATAL) << "No element name has been given";
				write(fs, fs.elname, c.cmat);
				if (fs.state & cv::FileStorage::INSIDE_MAP)
					fs.state = cv::FileStorage::NAME_EXPECTED + cv::FileStorage::INSIDE_MAP;
				return fs;
			}

			CHAOS_API friend std::ostream& operator<<(std::ostream& stream, const ConfusionMat& confusion);

			cv::SparseMat cmat; // NOC x NOC
			int noc;
		};

		CHAOS_API double GetMAP(const std::vector<ConfusionTable>& tables, const std::set<int>& idx = std::set<int>());
		CHAOS_API double GetAUC(const std::vector<ConfusionTable>& tables, const std::set<int>& idx = std::set<int>());

		CHAOS_API Mat GetFScore(const std::vector<ConfusionTable>& tables, double beta = 1, const std::set<int>& idx = std::set<int>());

		CHAOS_API Mat GetTPR(const std::vector<ConfusionTable>& tables, const std::set<int>& idx = std::set<int>());
		CHAOS_API Mat GetFPR(const std::vector<ConfusionTable>& tables, const std::set<int>& idx = std::set<int>());
		CHAOS_API Mat GetPPV(const std::vector<ConfusionTable>& tables, const std::set<int>& idx = std::set<int>());
		CHAOS_API Mat GetNPV(const std::vector<ConfusionTable>& tables, const std::set<int>& idx = std::set<int>());

	}
}