#pragma once

#include "highgui/highgui.hpp"

namespace chaos
{
	class CHAOS_API ScatterFigure : public Coordinate2D
	{
	public:
		ScatterFigure(const std::string& name);
		virtual ~ScatterFigure();

		template<class ... Args>
		void Scatter(const std::vector<Point>& points, Args ... args)
		{
			local = true;
			DummyWrap(Unpack(args)...);
			local = false;

			Apply(points);
		}

		static Ptr<ScatterFigure> Figure(const std::string& name = std::string());
	protected:
		virtual void Apply(const std::vector<Point>& points) = 0;
	};

	template<class ... Args>
	void Coordinate2D::Scatter(const std::vector<Point>& points, Args ... args)
	{
		auto aux = ScatterFigure::Figure();
		aux->Scatter(points, args...);
		aux->Set("XRange", x_range, "YRange", y_range, "Loose", 0.);

		aux->DrawOn(figure); 
	}
}