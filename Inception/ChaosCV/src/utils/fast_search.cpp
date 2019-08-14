#include "utils/fast_search.hpp"

#pragma warning (push, 0)
#include <faiss/IndexFlat.h>
#pragma warning (pop)

namespace chaos
{
	class Flat : public FastSearcher
	{
	public:
		Flat(int64 dims, const Method& method) : dims((int)dims)
		{
			index = faiss::IndexFlat(dims, (faiss::MetricType)method);
		}

		void Add(const dnn::Tensor& data) final
		{
			CHECK(data.IsContinue());
			CHECK_EQ(2, data.dims);
			CHECK_EQ(F32, data.depth);
			CHECK_EQ(dims, data.shape[1]);
			
			index.add(data.shape[0], (float*)data.data);
		}

		void Search(const dnn::Tensor& data, int k, dnn::Tensor& distances, dnn::Tensor& labels)
		{
			CHECK(data.IsContinue());
			CHECK_EQ(2, data.dims);
			CHECK_EQ(F32, data.depth);
			CHECK_EQ(dims, data.shape[1]);

			distances = dnn::Tensor({ data.shape[0], k }, F32);
			labels = dnn::Tensor({ data.shape[0], k }, S64);

			index.search(data.shape[0], (float*)data.data, k, (float*)distances.data, (int64*)labels.data);
		}

	private:
		int dims;
		faiss::IndexFlat index;
	};

	Ptr<FastSearcher> FastSearcher::CreateFlat(int dims, const Method& method)
	{
		return Ptr<FastSearcher>(new Flat(dims, method));
	}
}