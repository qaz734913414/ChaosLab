#pragma once

#include "highgui/highgui.hpp"

namespace chaos
{
	class CHAOS_API PlotFigure : public Coordinate2D
	{
	public:
		PlotFigure(const std::string& name);
		virtual ~PlotFigure();

		template<class ... Args>
		void Plot(const Mat& y, Args ... args)
		{
			local = true;
			DummyWrap(Unpack(args)...);
			local = false;

			Mat x = LineSpace(0.f, y.cols - 1.f, y.cols);
			Apply(x, y);
		}

		template<class ... Args>
		void Plot(const Mat& x, const Mat& y, Args ... args)
		{
			local = true;
			DummyWrap(Unpack(args)...);
			local = false;

			Apply(x, y);
		}

		static Ptr<PlotFigure> Figure(const std::string& name = std::string());
	protected:
		virtual void Apply(const Mat& x, const Mat& y) = 0;
	};

	template<class ... Args>
	void Coordinate2D::Plot(const Mat& x, const Mat& y, Args ... args)
	{
		auto aux = PlotFigure::Figure();
		aux->Plot(x, y, args...);
		aux->Set("XRange", x_range, "YRange", y_range);

		aux->DrawOn(figure);
	}

} // namespace chaos