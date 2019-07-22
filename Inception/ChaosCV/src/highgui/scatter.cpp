#include "highgui/scatter.hpp"

namespace chaos
{
	ScatterFigure::ScatterFigure(const std::string& name) : Coordinate2D(name) {}
	ScatterFigure::~ScatterFigure() {}

	class ScatterImpl : public ScatterFigure
	{
	public:
		ScatterImpl(const std::string& name) : ScatterFigure(name) {}
		~ScatterImpl() {}

		void Show() final
		{
			figure = ColorPool::Get(WHITE);

			ShowTags(0.3);

			ShowScatter();

			cv::rectangle(figure, roi, ColorPool::Get(BLACK));

			ColorPool::Reset();
			cv::imshow(name, figure);
		}

		Mat Draw() final
		{
			figure = ColorPool::Get(WHITE);

			ShowTags(0.3);

			ShowScatter();

			cv::rectangle(figure, roi, ColorPool::Get(BLACK));

			ColorPool::Reset();
			return figure;
		}

		void Apply(const std::vector<Point>& points) final
		{
			if (state == OFF) Clear();

			new_color = new_color == Color(-1) ? ColorPool::Next() : new_color;

			pts_color.push_back(new_color);
			pts_radius.push_back(new_radius);
			pts_marker.push_back(new_marker);
			pts_legend.push_back(new_legend);

			for (auto pt : points)
			{
				x_range[0] = pt.x < x_range[0] ? pt.x : x_range[0];
				x_range[1] = pt.x > x_range[1] ? pt.x : x_range[1];

				y_range[0] = pt.y < y_range[0] ? pt.y : y_range[0];
				y_range[1] = pt.y > y_range[1] ? pt.y : y_range[1];
			}
			pts_data.push_back(points);

			// Reset new args
			new_color = Color(-1);
			new_radius = 1;
			new_marker = -1;
			new_legend = "";
		}

	private:

		void Clear()
		{
			x_range[0] = y_range[0] = FLT_MAX;
			x_range[1] = y_range[1] = -FLT_MAX; //FLT_MIN;

			std::vector<std::vector<Point>>().swap(pts_data);

			std::vector<Color>().swap(pts_color);
			std::vector<int>().swap(pts_radius);
			std::vector<int>().swap(pts_marker);
			std::vector<std::string>().swap(pts_legend);
		}

		void ShowScatter()
		{
			Mat scatter = figure(roi);
			for (int n = 0; n < (int)pts_data.size(); n++)
			{
				for (auto point : pts_data[n])
				{
					float x_value = MappingX(point.x);
					float y_value = MappingY(point.y);

					cv::Point2i position = Point(x_value, y_value);
					switch (pts_marker[n])
					{
					case MARKER_CIRCLE:
						cv::circle(scatter, position, pts_radius[n], pts_color[n], pts_radius[n], cv::LINE_AA);
						break;
					case MARKER_CROSS:
					case MARKER_TILTED_CROSS:
					case MARKER_STAR:
					case MARKER_DIAMOND:
					case MARKER_SQUARE:
					case MARKER_TRIANGLE_UP:
					case MARKER_TRIANGLE_DOWN:
						cv::drawMarker(scatter, position, pts_color[n], pts_marker[n], pts_radius[n], pts_radius[n], cv::LINE_AA);
						break;
					default:
						cv::circle(scatter, position, 4, pts_color[n], -1, cv::LINE_AA);
						break;
					}
				}
			}
		}


		void ShowLegends()
		{
			float font_scale = 0.4f;
			auto font_face = cv::HersheyFonts::FONT_ITALIC;

			// Legend roi 
			auto legend = figure(Rect(5.f, roi.y, 70.f, roi.height / 2.f));

			for (int n = 0, k = 0; n < (int)pts_legend.size(); n++)
			{
				if (pts_legend[n].empty())
					continue;

				auto text_size = cv::getTextSize(pts_legend[n], font_face, font_scale, 1, 0); // always [X x 9]
				if (text_size.width + 16.f > 70)
				{
					LOG(WARNING) << "Legend \"" << pts_legend[n] << "\" is too long to show";
				}

				// Marker location
				Point mpt = Point(8.f, k * 24.f + 7.f);
				// Font location
				Point fpt = Point(16.f, k * 24.f + 12.f);

				if (fpt.y > 240)
				{
					LOG(WARNING) << "Too many legends to show: " << pts_legend[n];
				}

				// Draw marker
				switch (pts_marker[n])
				{
				case MARKER_CROSS:
				case MARKER_TILTED_CROSS:
				case MARKER_STAR:
				case MARKER_DIAMOND:
				case MARKER_SQUARE:
				case MARKER_TRIANGLE_UP:
				case MARKER_TRIANGLE_DOWN:
					cv::drawMarker(legend, mpt, pts_color[n], pts_marker[n], 6, 1, cv::LINE_AA);
					break;
				case MARKER_CIRCLE:
				default:
					cv::circle(legend, mpt, 3, pts_color[n], -1, cv::LINE_AA);
					break;
				}

				cv::putText(legend, pts_legend[n], fpt, font_face, font_scale, pts_color[n], 1, cv::LINE_AA);
				k++;
			}
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
				case "Radius"_hash:
					new_radius = std::any_cast<int>(arg_value);
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

		std::set<std::string> args_list = { "Radius", "Color", "Marker", "Legend", "Title", "XLabel", "YLabel", "XRange", "YRange" };

		std::vector<std::vector<Point>> pts_data;

		std::vector<Color> pts_color;
		std::vector<int> pts_radius;
		std::vector<int> pts_marker;
		std::vector<std::string> pts_legend;

		Color new_color = Color(-1);
		int new_radius = 1;
		int new_marker = -1;
		std::string new_legend = "";
	};

	Ptr<ScatterFigure> ScatterFigure::Figure(const std::string& name)
	{
		return Ptr<ScatterFigure>(new ScatterImpl(name));
	}
}