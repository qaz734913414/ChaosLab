#include "highgui/highgui.hpp"

namespace chaos
{
	std::vector<Color> CreateColorPool()
	{
		std::vector<Color> pool;
		// permutation produces 27 different colors
		std::vector<uchar> val = { 0, 0, 0, 128, 128, 128, 255, 255, 255 };
		do
		{
			auto same = std::find_if(pool.begin(), pool.end(), [=](Color& c) {
				if (c[0] == val[2] && c[1] == val[1] && c[2] == val[0])
				{
					return true;
				}
				return false;
				});
			if (same == pool.end())
			{
				pool.push_back(Color(val[2], val[1], val[0]));
			}
		} while (std::next_permutation(val.begin(), val.end()));
		return pool;
	}

	std::vector<Color> ColorPool::pool = CreateColorPool();
	int ColorPool::idx = 0;
	Color ColorPool::Get(int idx)
	{
		idx %= 27;
		return pool[idx];
	}
	Color ColorPool::Next()
	{
		Color color = pool[idx % 25 + (size_t)1];
		idx += 2;
		return color;
	}
	void ColorPool::Reset()
	{
		idx = 0;
	}



	int Figure::idx = 0;
	Figure::Figure(const std::string& _name) : IndefiniteParameter()
	{
		name = _name.empty() ? "Figure" + std::to_string(idx++) : _name;

		figure = Mat(600, 800, CV_8UC3, ColorPool::Get(WHITE));
	}
	Figure::~Figure() {}

	void Figure::Hold(State s)
	{
		state = s;
	}

	Coordinate2D::Coordinate2D(const std::string& name) : Figure(name)
	{
		x_range = Range(FLT_MAX, -FLT_MAX);
		y_range = Range(FLT_MAX, -FLT_MAX);
	}
	Coordinate2D::~Coordinate2D() {}

	float Coordinate2D::MappingX(float value)
	{
		return (value - x_range[0]) / (x_range[1] - x_range[0]) * (roi.width - 1.f);
	}
	float Coordinate2D::MappingY(float value)
	{
		return (roi.height - 1.f) - (value - y_range[0]) / (y_range[1] - y_range[0]) * (roi.height - 1.f);
	}

	void Coordinate2D::ShowTags()
	{
		// Common params for font
		auto font_face = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
		float font_scale = 0.3f;

		// If the range is very short, reset the range to 1
		if ((double)x_range[1] - x_range[0] < 1e-8) x_range = Range(x_range[0] - 0.5f, x_range[1] + 0.5f);
		if ((double)y_range[1] - y_range[0] < 1e-8) y_range = Range(y_range[0] - 0.5f, y_range[1] + 0.5f);

		// Scale
		double x_scale = pow(10, floor(log10(std::max(abs(x_range[1]), abs(x_range[0])))));
		double y_scale = pow(10, floor(log10(std::max(abs(y_range[1]), abs(y_range[0])))));

		float zero_x = MappingX(0.f);
		float zero_y = MappingY(0.f);
		cv::line(figure(roi), Point(0.f, zero_y), Point(roi.width, zero_y), ColorPool::Get(GRAY)); // X Axis
		cv::line(figure(roi), Point(zero_x, 0.f), Point(zero_x, roi.height), ColorPool::Get(GRAY)); // Y Axis

		cv::Mat x_space = LineSpace(x_range[0], x_range[1], index_cnt);
		cv::Mat y_space = LineSpace(y_range[0], y_range[1], index_cnt);
		auto x_axis = x_space.ptr<float>();
		auto y_axis = y_space.ptr<float>();

		float index_x;
		float index_y;
		// Show axis
		for (int i = 0; i < index_cnt; i++)
		{
			// X axis
			index_x = MappingX(x_axis[i]) + roi.tl().x;
			index_y = roi.br().y - 1.f;

			if (i % 5 == 0)
			{
				cv::line(figure, Point(index_x, index_y - index_len), Point(index_x, index_y + index_len), ColorPool::Get(BLACK));
				// scale value
				auto text = cv::format("%.2f", x_axis[i] / x_scale);
				auto text_size = cv::getTextSize(text, font_face, font_scale, 1, 0);
				cv::putText(figure, text, Point(index_x - text_size.width / 2.f, index_y + text_size.height + index_len + 2.f),
					font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
			}
			else
			{
				cv::line(figure, Point(index_x, index_y - index_len), Point(index_x, index_y), ColorPool::Get(BLACK));
			}

			// Y axis
			index_x = (float)roi.tl().x;
			index_y = MappingY(y_axis[i]) + roi.tl().y;

			if (i % 5 == 0)
			{
				cv::line(figure, Point(index_x - index_len, index_y), Point(index_x + index_len, index_y), ColorPool::Get(BLACK));
				// scale value
				auto text = cv::format("%.2f", y_axis[i] / y_scale);
				auto text_size = cv::getTextSize(text, font_face, font_scale, 1, 0);
				cv::putText(figure, text, Point(index_x - text_size.width - index_len - 2.f, index_y + text_size.height / 2.f),
					font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
			}
			else
			{
				cv::line(figure, Point(index_x, index_y), Point(index_x + index_len, index_y), ColorPool::Get(BLACK));
			}
		}

		auto offset = cv::getTextSize("-0.00", font_face, font_scale, 1, 0);
		// X Label and Scale
		if (x_scale != 1)
		{
			auto text = cv::format("%1.0e", x_scale);
			auto text_size = cv::getTextSize(text, font_face, font_scale, 1, 0);
			cv::putText(figure, text, Point(roi.br().x - text_size.width, roi.br().y + text_size.height + offset.height + index_len + 7),
				font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
		}
		if (!x_label.empty())
		{
			float font_scale = 0.4f;
			auto text_size = cv::getTextSize(x_label, font_face, font_scale, 1, 0);
			cv::putText(figure, x_label, Point(roi.x + roi.width / 2.f - text_size.width / 2.f, roi.br().y + text_size.height + offset.height + index_len + 17.f),
				font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
		}

		// Y Label and Scale
		cv::rotate(figure, figure, cv::ROTATE_90_CLOCKWISE);
		if (y_scale != 1)
		{
			auto text = cv::format("%1.0e", y_scale);
			auto text_size = cv::getTextSize(text, font_face, font_scale, 1, 0);
			cv::putText(figure, text, Point(figure.cols - roi.y - text_size.width, roi.x - offset.width - index_len - 7),
				font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
		}
		if (!y_label.empty())
		{
			float font_scale = 0.4f;
			auto text_size = cv::getTextSize(y_label, font_face, font_scale, 1, 0);
			cv::putText(figure, y_label, Point(figure.cols - roi.y - roi.height / 2.f - text_size.width / 2.f, roi.x - offset.width - index_len - 17.f),
				font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
		}
		cv::rotate(figure, figure, cv::ROTATE_90_COUNTERCLOCKWISE);

		// Show Title
		if (!title.empty())
		{
			float font_scale = 0.6f;
			auto text_size = cv::getTextSize(title, cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, font_scale, 1, 0);
			cv::putText(figure, title, cv::Point2f(figure.cols / 2.f - text_size.width / 2.f, roi.y / 2.f),
				font_face, font_scale, ColorPool::Get(BLACK), 1, cv::LINE_AA);
		}
	}
}