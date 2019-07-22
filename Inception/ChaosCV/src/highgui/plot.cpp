#include "highgui/plot.hpp"

namespace chaos
{
	PlotFigure::PlotFigure(const std::string& name) : Coordinate2D(name) {}
	PlotFigure::~PlotFigure() {}

	class PlotImpl : public PlotFigure
	{
	public:
		PlotImpl(const std::string& name) : PlotFigure(name) {}
		~PlotImpl() {}

		void Show() final
		{
			figure = ColorPool::Get(WHITE);

			ShowTags();

			ShowLegends();
			ShowCurves();

			cv::rectangle(figure, roi, ColorPool::Get(BLACK));

			ColorPool::Reset();

			cv::namedWindow(name);
			cv::setMouseCallback(name, OnMouse, this);
			cv::imshow(name, figure);
		}

		Mat Draw() final
		{
			figure = ColorPool::Get(WHITE);

			ShowTags();

			ShowLegends();
			ShowCurves();

			cv::rectangle(figure, roi, ColorPool::Get(BLACK));

			ColorPool::Reset();

			return figure;
		}

		void Apply(const Mat& x, const Mat& y) final
		{
			CHECK_EQ(1, x.rows);
			CHECK_EQ(1, y.rows);
			CHECK_EQ(x.cols, y.cols);
			CHECK_EQ(x.type(), CV_32FC1);
			CHECK_EQ(y.type(), CV_32FC1);

			if (state == OFF) Clear();

			new_color = new_color == Color(-1) ? ColorPool::Next() : new_color;

			curves_color.push_back(new_color);
			curves_width.push_back(new_width);
			curves_marker.push_back(new_marker);
			curves_legend.push_back(new_legend);

			// Update the range of axis
			UpdateRange(x, y);

			x_data.push_back(x);
			y_data.push_back(y);

			// Reset new args
			new_color = Color(-1);
			new_width = 1;
			new_marker = -1;
			new_legend = "";
		}

	private:
		void Clear()
		{
			x_range[0] = y_range[0] = FLT_MAX;
			x_range[1] = y_range[1] = -FLT_MAX; //FLT_MIN;

			std::vector<Mat>().swap(x_data);
			std::vector<Mat>().swap(y_data);

			std::vector<Color>().swap(curves_color);
			std::vector<int>().swap(curves_width);
			std::vector<int>().swap(curves_marker);
			std::vector<std::string>().swap(curves_legend);
		}

		void ShowCurves()
		{
			Mat curve = figure(roi);
			// Show curve in roi area
			for (int n = 0; n < (int)y_data.size(); n++)
			{
				auto x_ptr = x_data[n].ptr<float>();
				auto y_ptr = y_data[n].ptr<float>();

				std::vector<cv::Point2i> pts;
				for (int i = 0; i < y_data[n].size().width; i++)
				{
					if (isnan(x_ptr[i]) || isnan(y_ptr[i]) || isinf(x_ptr[i]) || isinf(y_ptr[i]))
					{
						if (!pts.empty())
						{
							// Here just support Point2i in newer version
							cv::polylines(curve, pts, false, curves_color[n], curves_width[n], cv::LINE_AA);
							std::vector<cv::Point2i>().swap(pts);
						}
						continue;
					}

					float x_value = MappingX(x_ptr[i]);
					float y_value = MappingY(y_ptr[i]);

					pts.push_back(Point(x_value, y_value));

					switch (curves_marker[n])
					{
					case MARKER_CIRCLE:
						cv::circle(curve, pts.back(), curves_width[n] * 4, curves_color[n], curves_width[n], cv::LINE_AA);
						break;
					case MARKER_CROSS:
					case MARKER_TILTED_CROSS:
					case MARKER_STAR:
					case MARKER_DIAMOND:
					case MARKER_SQUARE:
					case MARKER_TRIANGLE_UP:
					case MARKER_TRIANGLE_DOWN:
						cv::drawMarker(curve, pts.back(), curves_color[n], curves_marker[n], curves_width[n] * 8, curves_width[n], cv::LINE_AA);
						break;
					default:
						break;
					}
				}
				// Here just support Point2i in newer version
				cv::polylines(curve, pts, false, curves_color[n], curves_width[n], cv::LINE_AA);
			}
		}

		void ShowLegends()
		{
			float font_scale = 0.4f;
			auto font_face = cv::HersheyFonts::FONT_ITALIC;

			// Legend roi 
			auto legend = figure(Rect(5.f, roi.y, 70.f, roi.height / 2.f));

			for (int n = 0, k = 0; n < (int)curves_legend.size(); n++)
			{
				if (curves_legend[n].empty())
					continue;

				auto text_size = cv::getTextSize(curves_legend[n], font_face, font_scale, 1, 0); // always [X x 9]
				if (text_size.width + 16.f > 70)
				{
					LOG(WARNING) << "Legend \"" << curves_legend[n] << "\" is too long to show";
				}

				// Marker location
				Point mpt = Point(8.f, k * 24.f + 7.f);
				// Font location
				Point fpt = Point(16.f, k * 24.f + 12.f);

				if (fpt.y > 240)
				{
					LOG(WARNING) << "Too many legends to show: " << curves_legend[n];
				}

				// Draw marker
				switch (curves_marker[n])
				{
				case MARKER_CROSS:
				case MARKER_TILTED_CROSS:
				case MARKER_STAR:
				case MARKER_DIAMOND:
				case MARKER_SQUARE:
				case MARKER_TRIANGLE_UP:
				case MARKER_TRIANGLE_DOWN:
					cv::drawMarker(legend, mpt, curves_color[n], curves_marker[n], 6, 1, cv::LINE_AA);
					break;
				case MARKER_CIRCLE:
				default:
					cv::circle(legend, mpt, 3, curves_color[n], -1, cv::LINE_AA);
					break;
				}

				cv::putText(legend, curves_legend[n], fpt, font_face, font_scale, curves_color[n], 1, cv::LINE_AA);
				k++;
			}
		}

		void UpdateRange(const Mat& x, const Mat& y)
		{
			// Update the range of axis
			double min_x, max_x;
			cv::minMaxIdx(x, &min_x, &max_x, nullptr, nullptr, x != INFINITY & x != -INFINITY);
			CHECK(isfinite(min_x) && isfinite(max_x));

			x_range[0] = min_x < x_range[0] ? (float)min_x : x_range[0];
			x_range[1] = max_x > x_range[1] ? (float)max_x : x_range[1];

			double min_y, max_y;
			cv::minMaxIdx(y, &min_y, &max_y, nullptr, nullptr, y != INFINITY & y != -INFINITY);
			CHECK(isfinite(min_y) && isfinite(max_y));

			y_range[0] = min_y < y_range[0] ? (float)min_y : y_range[0];
			y_range[1] = max_y > y_range[1] ? (float)max_y : y_range[1];
		}

		static void OnMouse(int event, int x, int y, int flags, void* user_data)
		{
			auto plot = static_cast<PlotImpl*>(user_data);

			if (plot->figure.empty()) return;

			cv::Point pt(x, y);

			cv::Mat on_mouse = plot->figure.clone();

			if (pt.inside(plot->roi))
			{
				auto font_face = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
				float font_scale = 0.35f;

				// Inverse mapping
				auto mouse_value = (x - plot->roi.x) * (plot->x_range[1] - plot->x_range[0]) / (plot->roi.width - 1.f) + plot->x_range[0];

				std::vector<int> idx;
				// Search nearest point idx
				for (auto x_value : plot->x_data)
				{
					int min_idx[2]; // 1
					cv::minMaxIdx(cv::abs(x_value - mouse_value), nullptr, nullptr, min_idx);
					idx.push_back(min_idx[1]);
				}

				for (size_t i = 0; i < idx.size(); i++)
				{
					float x_value = plot->x_data[i].at<float>(0, idx[i]);
					float y_value = plot->y_data[i].at<float>(0, idx[i]);

					float xpt = plot->MappingX(x_value);
					float ypt = plot->MappingY(y_value);

					// aux line
					//cv::line(on_mouse(plot->roi), Point(xpt, 0.f), Point(xpt, plot->roi.height), Color(96, 96, 96));
					//cv::line(on_mouse(plot->roi), Point(0.f, ypt), Point(plot->roi.width, ypt), Color(96, 96, 96));

					cv::circle(on_mouse(plot->roi), Point(xpt, ypt), 3, plot->curves_color[i], -1);

					// Show Text
					std::string value = cv::format("(%.3g,%.3g)", x_value, y_value);
					auto text_size = cv::getTextSize(value, font_face, font_scale, 1, 0); // always [X x 8]

					Point vpt = Point(10.f, i * 24.f + 322.f);
					if (vpt.y > 540)
						continue;

					// Translucent
					on_mouse(Rect(vpt.x, vpt.y - 15.f, (float)text_size.width, 24.f)) += Color(180, 180, 180);
					cv::putText(on_mouse, value, vpt, font_face, font_scale, plot->curves_color[i], 1, cv::LINE_AA);
				}
			}

			cv::imshow(plot->name, on_mouse);
		}

		virtual void Parse(const std::any& any) final
		{
			if (any.type() == typeid(const char*) && args_list.find(std::any_cast<const char*>(any)) != args_list.end())
			{
				const char* arg = std::any_cast<const char*>(any);
				local ? ParseLoacl(arg) : ParseGlobal(arg);
			}
			else
			{
				arg_value = any;
			}
		}
		virtual void ParseLoacl(const char* arg) final
		{
			try
			{
				switch (Hash(arg))
				{
				case "LineWidth"_hash:
					new_width = std::any_cast<int>(arg_value);
					break;
				case "Color"_hash:
					new_color = std::any_cast<Color>(arg_value);
					break;
				case "Marker"_hash:
					new_marker = std::any_cast<MarkerTypes>(arg_value);
					break;
				case "Legend"_hash:
					new_legend = std::any_cast<const char*>(arg_value);
					break;
				default:
					LOG(WARNING) << "Unknown arg " << arg;
					break;
				}
			}
			catch (std::bad_any_cast err)
			{
				LOG(FATAL) << arg << " cast error " << err.what();
			}
		}
		virtual void ParseGlobal(const char* arg) final
		{
			try
			{
				switch (Hash(arg))
				{
				case "Title"_hash:
					title = std::any_cast<const char*>(arg_value);
					break;
				case "XLabel"_hash:
					x_label = std::any_cast<const char*>(arg_value);
					break;
				case "YLabel"_hash:
					y_label = std::any_cast<const char*>(arg_value);
					break;
				case "XRange"_hash:
					x_range = std::any_cast<Range>(arg_value);
					break;
				case "YRange"_hash:
					y_range = std::any_cast<Range>(arg_value);
					break;
				default:
					LOG(WARNING) << "Unknown arg " << arg;
					break;
				}
			}
			catch (std::bad_any_cast err)
			{
				LOG(FATAL) << arg << " cast error " << err.what();
			}
		}

		std::set<std::string> args_list = { "LineWidth", "Color", "Marker", "Legend", "Title", "XLabel", "YLabel", "XRange", "YRange" };

		std::vector<Mat> y_data;
		std::vector<Mat> x_data;

		std::vector<Color> curves_color;
		std::vector<int> curves_width;
		std::vector<int> curves_marker;
		std::vector<std::string> curves_legend;

		Color new_color = Color(-1);
		int new_width = 1;
		int new_marker = -1;
		std::string new_legend = "";
	};

	Ptr<PlotFigure> PlotFigure::Figure(const std::string& name)
	{
		return Ptr<PlotFigure>(new PlotImpl(name));
	}
}