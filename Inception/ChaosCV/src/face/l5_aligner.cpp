#include "face/aligner.hpp"

namespace chaos
{
	namespace face
	{
		class L5 : public Aligner
		{
		public:
			L5(const cv::Size& size, float pad, const cv::Point2f& offset) : size(size)
			{
				const Landmark relative_points = {
					{ 30.2946f / 96.f, 51.6963f / 112.f },
					{ 65.5318f / 96.f, 51.5014f / 112.f },
					{ 48.0252f / 96.f, 71.7366f / 112.f },
					{ 33.5493f / 96.f, 92.3655f / 112.f },
					{ 62.7299f / 96.f, 92.2041f / 112.f }
				};

				float scale = size.height > size.width ? size.width / 96.f : size.height / 112.f;
				int diff = std::abs(size.height - size.width);

				Size face_size = Size(96, 112) * scale;
				auto bias = offset + ((Point(size) - Point(face_size)) / 2.f); // offset

				for (size_t i = 0; i < relative_points.size(); ++i)
				{
					auto pt = (Point(pad, pad) + relative_points[i]) / (2.f * pad + 1);
					target_points.push_back(Point(face_size.width * pt.x + bias.x,
						face_size.height * pt.y + bias.y));
				}
			}

			Mat Align(const Mat& image, const Landmark& points) final
			{
				Mat T_inv;
				Mat M = FindSimilarityTransform(points, target_points, T_inv);

				Mat face;
				cv::warpAffine(image, face, M, size);

				return face;
			}

		private:
			Landmark target_points;
			cv::Size size;
		};

		Ptr<Aligner> Aligner::CreateL5(const Size& size, float pad, const Point& offset)
		{
			return Ptr<Aligner>(new L5(size, pad, offset));
		}
	}
}