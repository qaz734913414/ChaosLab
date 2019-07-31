#include "test/cumulative.hpp"

namespace chaos
{
	namespace test
	{
		CumulativeTabel::CumulativeTabel() {}

		CumulativeTabel::CumulativeTabel(int noc)
		{
			table = Mat::zeros(1, noc, CV_32FC1);
		}

		CumulativeTabel::CumulativeTabel(const Mat& data) : table(data) {}

		void CumulativeTabel::Apply(int actual_id, const std::vector<double>& prob)
		{
			std::multimap<double, int> sorted;
			for (int i = 0; i < (int)prob.size(); i++)
			{
				sorted.insert(std::make_pair(1. - prob[i], i));
			}

			int r = 0;
			for (auto data : sorted)
			{
				if (data.second == actual_id)
				{
					table(cv::Rect(r, 0, table.cols - r, 1)) += 1;
					break;
				}
				r++;
			}
		}

		Mat CumulativeTabel::GetPrecision() const
		{
			int num = (int)table.ptr<float>()[table.cols - 1];
			return table / num;
		}

		CumulativeTabel& CumulativeTabel::operator+(const CumulativeTabel& c)
		{
			CHECK(!table.empty());
			CHECK_EQ(table.cols, c.table.cols); // same noc

			table += c.table;
			return *this;
		}
	}
}