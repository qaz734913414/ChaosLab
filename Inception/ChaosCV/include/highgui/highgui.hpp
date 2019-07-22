#pragma once

#include "core/core.hpp"
#include "utils/utils.hpp"

namespace chaos
{
	using Color = Scalar;

	enum ColorName
	{
		BLACK = 0,
		BLUE = 2,
		CYAN = 8,
		GRAY = 13,
		GREEN = 3,
		LIME = 6,
		MAGENTA = 20,
		MAROON = 9,
		NAVY = 1,
		OLIVE = 12,
		PURPLE = 10,
		RED = 18,
		TEAL = 4,
		WHITE = 26,
		YELLOW = 24,
	};

	/// <summary>
	/// <para>27 colors pool</para>
	/// </summary>
	class CHAOS_API ColorPool
	{
	public:
		static Color Get(int idx);
		static Color Next();
		static void Reset();

	private:
		static std::vector<Color> pool;
		static int idx;
	};


	// Figure State
	enum State
	{
		OFF,
		ON,
	};

	/// <summary>Reference to OpenCV</summary>
	enum MarkerTypes
	{
		MARKER_CROSS = 0,           //!< A crosshair marker shape
		MARKER_TILTED_CROSS = 1,    //!< A 45 degree tilted crosshair marker shape
		MARKER_STAR = 2,            //!< A star marker shape, combination of cross and tilted cross
		MARKER_DIAMOND = 3,         //!< A diamond marker shape
		MARKER_SQUARE = 4,          //!< A square marker shape
		MARKER_TRIANGLE_UP = 5,     //!< An upwards pointing triangle marker shape
		MARKER_TRIANGLE_DOWN = 6,   //!< A downwards pointing triangle marker shape
		// Extension
		MARKER_CIRCLE = 7,          //!< A circle marker shape
	};



	class CHAOS_API Figure : public IndefiniteParameter
	{
	public:
		Figure(const std::string& name);
		virtual ~Figure();

		void Hold(State state);
		virtual void Show() = 0;
		virtual Mat Draw() = 0;

	protected:
		static int idx;
		std::string name;
		std::string title;

		State state;

		Mat figure;
		// 640 x 480 from (140, 60)
		const Rect roi = Rect(140, 60, 640, 480);;

		bool local = false;
	};

	class CHAOS_API Coordinate2D : public Figure
	{
	protected:
		Coordinate2D(const std::string& name);
		virtual ~Coordinate2D();

		float MappingX(float value);
		float MappingY(float value);

		/// <summary>To show the axis and labels</summary>
		virtual void ShowTags();

		Range x_range;
		Range y_range;

		std::string x_label;
		std::string y_label;

		const int index_cnt = 51;
		const int index_len = 5;
	};
}