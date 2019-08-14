#pragma once

#include "core/core.hpp"
#include "dnn/tensor.hpp"

namespace chaos
{
	// Use Faiss to fast search
	class CHAOS_API FastSearcher
	{
	public:
		/// <summary>Distance Method</summary>
		enum Method
		{
			IP,		///<summary>maximum inner product search</summary>
			L2,		///<summary>squared L2 search</summary>
			L1,		///<summary>L1 (aka cityblock)</summary>
			LINF,	///<summary>infinity distance</summary>
			//LP,	///<summary> L_p distance, p is given by metric_arg</summary>
		};

		virtual void Add(const dnn::Tensor& data) = 0;

		virtual void Search(const dnn::Tensor& data, int k, dnn::Tensor& distances, dnn::Tensor& labels) = 0;

		static Ptr<FastSearcher> CreateFlat(int dims, const Method& method = L2);
	};
}