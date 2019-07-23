#pragma once

#include "face/face_info.hpp"

namespace chaos
{
	namespace face
	{
		class CHAOS_API Aligner
		{
		public:
			~Aligner() {}

			virtual Mat Align(const Mat& image, const Landmark& points) = 0;

			/// <summary>
			/// <para>SphereFace 5-points face aligner</para>
			/// </summary>
			/// <param name="size">Target size</param>
			/// <param name="pad">Padding, refer to dlib</param>
			/// <param name="offset">Offset</param>
			static Ptr<Aligner> CreateL5(const Size& size, float pad = 0.f, const Point & offset = Point());
		};
	}
}