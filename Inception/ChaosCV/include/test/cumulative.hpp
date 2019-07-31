#pragma once

#include "core/core.hpp"

namespace chaos
{
	namespace test
	{
		class CHAOS_API CumulativeTabel
		{
		public:
			CumulativeTabel();
			CumulativeTabel(int noc);
			CumulativeTabel(const Mat& data);
			void Apply(int actual_id, const std::vector<double>& prob);

			Mat GetPrecision() const;

			CumulativeTabel& operator+(const CumulativeTabel& c);

			CHAOS_API friend void operator>>(const cv::FileNode& node, CumulativeTabel& table)
			{
				Mat data;
				cv::read(node, data, Mat());
				table = data;
			}

			CHAOS_API friend cv::FileStorage& operator<<(cv::FileStorage& fs, const CumulativeTabel& c)
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

		private:
			Mat table; // 1 x NOC;
		};
	}
}