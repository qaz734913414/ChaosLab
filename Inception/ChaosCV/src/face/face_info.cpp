#include "face/face_info.hpp"

namespace chaos
{
	namespace face
	{
		FaceInfo::FaceInfo() : score(0), rect(Rect()) {}
		FaceInfo::FaceInfo(const Rect& rect, float score) : rect(rect), score(score) {}
		FaceInfo::FaceInfo(const ObjectRect& obj) : score(obj.score), rect(obj.rect) {}

		void Sort(const Rect& center, std::vector<FaceInfo>& faces)
		{
			std::sort(faces.begin(), faces.end(), [=](const FaceInfo& f1, const FaceInfo& f2) {
				return (f1.rect & center).area() == (f2.rect & center).area() ?
					f1.rect.area() > f2.rect.area() : (f1.rect & center).area() > (f2.rect & center).area();
			});
		}
	}
}