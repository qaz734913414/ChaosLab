#pragma once

#include "face/face_info.hpp"
#include "dnn/net.hpp"

namespace chaos
{
	namespace face
	{
		class CHAOS_API Clusterer : public IndefiniteParameter
		{
		public:
			virtual ~Clusterer() {};

			virtual void Add(const dnn::Tensor& feat) = 0;
			virtual dnn::Tensor Cluster() = 0;

			static Ptr<Clusterer> LoadGCN(const dnn::Model& model, const dnn::Context& ctx = dnn::Context());
		};
	}
}