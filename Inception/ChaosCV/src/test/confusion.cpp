#include "test/confusion.hpp"

namespace chaos
{
	namespace test
	{
		ConfusionTable::ConfusionTable(int nbins) : nbins(nbins)
		{
			table = Mat::zeros(4, nbins, CV_32FC1);
		}
		ConfusionTable::ConfusionTable(const Mat& table) : table(table), nbins(table.cols) {}

		void ConfusionTable::Apply(bool is_positive, double prob)
		{
			int i = static_cast<int>(prob * (nbins - 1.));
			if (is_positive)
			{
				table(cv::Rect(0, 0, i + 1, 1)) += 1; //TP
				table(cv::Rect(i + 1, 2, nbins - i - 1, 1)) += 1; // FN
			}
			else
			{
				table(cv::Rect(0, 1, i + 1, 1)) += 1; // FP
				table(cv::Rect(i + 1, 3, nbins - i - 1, 1)) += 1; // TN
			}
		}

		Mat ConfusionTable::GetTPR() const
		{
			Mat tp = table.row(0);
			Mat fn = table.row(2);
			return tp / (tp + fn);
		}
		Mat ConfusionTable::GetFPR() const
		{
			Mat fp = table.row(1);
			Mat tn = table.row(3);
			return fp / (fp + tn);
		}
		Mat ConfusionTable::GetPPV() const
		{
			Mat tp = table.row(0);
			Mat fp = table.row(1);
			return (tp + FLT_EPSILON) / (tp + fp + FLT_EPSILON);
		}
		Mat ConfusionTable::GetNPV() const
		{
			Mat tn = table.row(3);
			Mat fn = table.row(2);
			return (tn + FLT_EPSILON) / (fn + tn + FLT_EPSILON);
		}

		Mat ConfusionTable::GetACC() const
		{
			Mat tp = table.row(0);
			Mat fn = table.row(2);
			Mat fp = table.row(1);
			Mat tn = table.row(3);

			return (tp + tn) / (tp + fn + fp + tn);
		}

		double ConfusionTable::GetAP() const
		{
			auto tpr = GetTPR();
			auto ppv = GetPPV();

			double ap = 0;
			auto x = tpr.ptr<float>();
			auto y = ppv.ptr<float>();
			for (int i = 1; i < nbins; i++)
			{
				if (isnan(x[i]) || isnan(y[i])) continue;

				auto delta_x = x[i - 1] - x[i];
				auto mean_y = 0.5 * y[i] + 0.5 * y[i - 1];

				ap += (delta_x * mean_y);
			}

			return ap;
		}

		double ConfusionTable::GetAUC() const
		{
			auto fpr = GetFPR();
			auto tpr = GetTPR();

			double auc = 0;
			auto x = fpr.ptr<float>();
			auto y = tpr.ptr<float>();
			for (int i = 1; i < nbins; i++)
			{
				if (isnan(x[i]) || isnan(y[i])) continue;

				auto delta_x = x[i - 1] - x[i];
				auto mean_y = 0.5 * y[i] + 0.5 * y[i - 1];

				auc += (delta_x * mean_y);
			}

			return auc;
		}

		Mat ConfusionTable::GetFScore(double beta) const
		{
			auto ppv = GetPPV();
			auto tpr = GetTPR();

			return (1 + beta * beta) * ppv.mul(tpr) / (beta * beta * ppv + tpr);
		}





		ConfusionMat::ConfusionMat() : noc(0) {}
		ConfusionMat::ConfusionMat(int noc) : noc(noc)
		{
			int shape[] = { noc, noc };
			cmat.create(2, shape, CV_32SC1);
		}
		ConfusionMat::ConfusionMat(const cv::SparseMat& _cmat) : cmat(_cmat)
		{
			CHECK_EQ(2, cmat.dims());
			CHECK_EQ(cmat.size()[0], cmat.size()[1]);
			noc = cmat.size()[0];
		}

		void ConfusionMat::Apply(int actual_id, int predict_id)
		{
			cmat.ref<int>(predict_id, actual_id)++;
		}

		double ConfusionMat::GetACC() const
		{
			double tp = 0, sum = 0;
			for (auto it = cmat.begin<int>(); it != cmat.end<int>(); ++it)
			{
				sum += it.value<int>();
				if (it.node()->idx[0] == it.node()->idx[1])
					tp += it.value<int>();
			}
			return tp / sum;
		}

		std::ostream& operator<<(std::ostream& stream, const ConfusionMat& confusion)
		{
			for (auto it = confusion.cmat.begin<int>(); it != confusion.cmat.end<int>(); ++it)
			{
				stream << "(" << it.node()->idx[0] << ", " << it.node()->idx[1] << ", " << it.value<int>() << ")" << std::endl;
			}
			return stream;
		}




		double GetMAP(const std::vector<ConfusionTable>& tables, const std::set<int>& idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			double mAP = 0;
			for (auto i : index)
			{
				mAP += tables[i].AP;
			}
			return mAP / index.size();
		}

		double GetAUC(const std::vector<ConfusionTable>& tables, const std::set<int>& idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			double auc = 0;
			for (auto i : index)
			{
				auc += tables[i].AUC;
			}
			return auc / index.size();
		}

		Mat GetFScore(const std::vector<ConfusionTable>& tables, double beta, const std::set<int> & idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			Mat FScore = Mat::zeros(tables[0].TPR.size(), tables[0].TPR.type());
			for (auto i : index)
			{
				auto local = tables[i].GetFScore(beta);
				cv::add(FScore, local, FScore, local == local);
			}
			return FScore / (double)index.size();
		}

		Mat GetTPR(const std::vector<ConfusionTable>& tables, const std::set<int>& idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			Mat TPR = Mat::zeros(tables[0].TPR.size(), tables[0].TPR.type());
			for (auto i : index)
			{
				auto local = tables[i].TPR;
				cv::add(TPR, local, TPR, local == local);
			}
			return TPR / (double)index.size();
		}
		Mat GetFPR(const std::vector<ConfusionTable>& tables, const std::set<int>& idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			Mat FPR = Mat::zeros(tables[0].FPR.size(), tables[0].FPR.type());
			for (auto i : index)
			{
				auto local = tables[i].FPR;
				cv::add(FPR, local, FPR, local == local);
			}
			return FPR / (double)index.size();
		}
		Mat GetPPV(const std::vector<ConfusionTable>& tables, const std::set<int>& idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			Mat PPV = Mat::zeros(tables[0].PPV.size(), tables[0].PPV.type());
			for (auto i : index)
			{
				auto local = tables[i].PPV;
				cv::add(PPV, local, PPV, local == local);
			}
			return PPV / (double)index.size();
		}
		Mat GetNPV(const std::vector<ConfusionTable>& tables, const std::set<int>& idx)
		{
			std::set<int> index = idx;
			if (index.empty())
			{
				for (int i = 0; i < (int)tables.size(); i++) index.insert(i);
			}

			Mat NPV = Mat::zeros(tables[0].NPV.size(), tables[0].NPV.type());
			for (auto i : index)
			{
				auto local = tables[i].NPV;
				cv::add(NPV, local, NPV, local == local);
			}
			return NPV / (double)index.size();
		}

	}
}