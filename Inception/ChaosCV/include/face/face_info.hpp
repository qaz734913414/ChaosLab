#pragma once

#include "core/core.hpp"
#include "utils/utils.hpp"

namespace chaos
{
	namespace face
	{
		using Landmark = std::vector<Point>;
		class CHAOS_API FaceInfo
		{
		public:
			FaceInfo();
			FaceInfo(const ObjectRect& obj);

			float score;
			Rect rect;
			Landmark points = Landmark();
		};

		void Sort(const Rect& center, std::vector<FaceInfo>& infos);
	}
}