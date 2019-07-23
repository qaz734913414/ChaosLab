#pragma once

#include "core/core.hpp"

namespace chaos
{
	enum IoUType
	{
		IOU_UNION,
		IOU_MIN,
		IOU_MAX
	};

	enum WeightType
	{
		WEIGHT_LINEAR,
		WEIGHT_GAUSSIAN,
		WEIGHT_ORIGINAL
	};

	class CHAOS_API ObjectRect
	{
	public:
		ObjectRect(const Rect& rect, float score);

		float score;
		Rect rect;
	};

	/// <summary>Refer to happynear</summary>
	CHAOS_API std::vector<int> SoftNMS(std::vector<ObjectRect>& objects, double overlap_rate, double min_confidence,
		IoUType iou_type = IOU_UNION, WeightType weight_type = WEIGHT_LINEAR);

	CHAOS_API Mat Crop(const Mat& src, const Rect& roi, const Size& size,
		int flags = cv::INTER_LINEAR, int border_type = cv::BORDER_CONSTANT, const Scalar& border_value = 0);

	CHAOS_API void MakeRectSquare(Rect& rect);

	CHAOS_API Mat LineSpace(float a, float b, int n);

	CHAOS_API Mat FindSimilarityTransform(std::vector<Point> source_points, std::vector<Point> target_points, Mat& T_inv);

}