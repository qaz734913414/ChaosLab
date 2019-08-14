#include "dnn/tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		inline int Cast(const Depth& depth)
		{
			switch (depth)
			{
			case F64:
				return CV_64F;
			case F32:
				return CV_32F;
			case S32:
				return CV_32S;
			case F16:
				return CV_16F;
			case S16:
				return CV_16S;
			case U16:
				return CV_16U;
			case S8:
				return CV_8S;
			case U8:
				return CV_8U;
			default:
				LOG(FATAL) << "DO not support to Convert OpenCV Mat depth";
				return -1; // Never Reachable
			}
		}

		inline Depth Cast(int depth)
		{
			switch (depth)
			{
			case CV_64F:
				return F64;
			case CV_32S:
				return S32;
			case CV_32F:
				return F32;
			case CV_16U:
				return U16;
			case CV_16S:
				return S16;
			case CV_16F:
				return F16;
			case CV_8S:
				return S8;
			case CV_8U:
				return U8;
			default:
				LOG(FATAL) << "Unknown depth";
				return U8; // Never Reachable
			}
		}

		inline std::string ToString(const Depth& depth)
		{
			switch (depth)
			{
			case F64:
				return "float64";
			case S64:
				return "int64";
			case F32:
				return "float32";
			case S32:
				return "int32";
			case S16:
				return "int16";
			case U16:
				return "uint32";
			case F16:
				return "float16";
			case U8:
				return "uint8";
			case S8:
				return "int8";
			default:
				LOG(FATAL) << "Unknown depth";
				return "unknown"; // Never Reachable
			}
		}




		Tensor::Tensor() : data(nullptr), ref_cnt(nullptr), allocator(nullptr), depth(U8), shape(Shape()), cstep(0), dims(0), aligned(false) {}

		Tensor::Tensor(const Shape& shape, const Depth& depth, bool aligned, Allocator* allocator) : Tensor()
		{
			Create(shape, depth, aligned, allocator);
		}
		Tensor::Tensor(const Shape& shape, const Depth& depth, void* data, bool aligned, Allocator* allocator) 
			: data(data), ref_cnt(nullptr), aligned(aligned), shape(shape), dims(static_cast<int>(shape.Size())), depth(depth), allocator(allocator)
		{
			//CHECK_NE(UNK, depth);
			cstep = dims >= 2 ? (size_t)shape[dims - 1LL] * shape[dims - 2LL] : shape[0];
			if (aligned) cstep = AlignSize(cstep * (depth >> DEPTH_SHIFT), MALLOC_ALIGN) / (depth >> DEPTH_SHIFT);
		}

		Tensor::Tensor(const Tensor& t) 
			: data(t.data), ref_cnt(t.ref_cnt), aligned(t.aligned), shape(t.shape), dims(t.dims), depth(t.depth), cstep(t.cstep), allocator(t.allocator)
		{
			if (ref_cnt)
				CHAOS_XADD(ref_cnt, 1);
		}

		Tensor::~Tensor()
		{
			Release();
		}

		Tensor& Tensor::operator=(const Tensor& t)
		{
			if (this == &t)
				return *this;

			if (t.ref_cnt)
				CHAOS_XADD(t.ref_cnt, 1);

			Release();

			data = t.data;
			ref_cnt = t.ref_cnt;
			aligned = t.aligned;
			shape = t.shape;
			dims = t.dims;
			depth = t.depth;
			cstep = t.cstep;
			allocator = t.allocator;

			return *this;
		}

		Tensor Tensor::Unroll(const std::vector<Mat>& vdata, bool rechannel, bool aligned, Allocator* allocator)
		{
			CHECK(!vdata.empty());

			Shape shape = { (int)vdata.size(), vdata[0].channels(), vdata[0].rows, vdata[0].cols };
			Tensor tensor = Tensor(shape, Cast(vdata[0].depth()), aligned, allocator);

			for (int n = 0; n < tensor.shape[0]; n++)
			{
				CHECK_EQ(2, vdata[n].dims);
				CHECK_EQ(vdata[0].depth(), vdata[n].depth());
				CHECK_EQ(shape[1], vdata[n].channels());
				CHECK_EQ(shape[2], vdata[n].rows);
				CHECK_EQ(shape[3], vdata[n].cols);

				std::vector<Mat> slice(tensor.shape[1]);
				int c = rechannel ? shape[1] - 1 : 0;
				for (int i = 0; i < shape[1]; i++)
				{
					int idx = std::abs(c - i);
					slice[idx] = Mat(vdata[n].size(), vdata[n].depth(), (uchar*)tensor.data + ((size_t)shape[1] * n + i) * tensor.cstep * vdata[n].elemSize1());
				}
				cv::split(vdata[n], slice);
			}

			return tensor;
		}

		std::vector<Mat> Tensor::Rollup(bool rechannel) const
		{
			CHECK_EQ(4, dims);

			std::vector<Mat> _data;
			for (int n = 0; n < shape[0]; n++)
			{
				Mat packed;
				std::vector<cv::Mat> slice(shape[1]);
				int c = rechannel ? shape[1] - 1 : 0;
				for (int i = 0; i < shape[1]; i++)
				{
					int idx = std::abs(c - i);
					slice[idx] = cv::Mat(shape[2], shape[3], Cast(depth), (uchar*)data + ((size_t)shape[1] * n + i) * cstep * (depth >> DEPTH_SHIFT));
				}
				cv::merge(slice, packed);

				_data.push_back(packed);
			}

			return _data;
		}

		Tensor::operator Mat() const
		{
			std::vector<size_t> steps;
			switch (dims)
			{
			default:
				for (int i = 1; i < dims - 2; i++)
				{
					size_t step = cstep * (depth >> DEPTH_SHIFT);
					for (int j = 0; j < dims - 2; j++)
					{
						step *= shape[j];
					}
					steps.push_back(step);
				}
			case 3:
				steps.push_back(cstep * (depth >> DEPTH_SHIFT));
			case 2:
				steps.push_back((size_t)shape.back() * (depth >> DEPTH_SHIFT));
			case 1:
				steps.push_back((depth >> DEPTH_SHIFT));
			}
			return Mat(dims, shape.data(), Cast(depth), data, steps.data());
		}

		void Tensor::Create(const Shape& _shape, const Depth& _depth, bool _aligned, Allocator* _allocator)
		{
			if (shape == _shape && depth == _depth && allocator == _allocator) return;

			Release();

			//CHECK_NE(UNK, _depth);

			shape = _shape;
			depth = _depth;
			aligned = _aligned;
			allocator = _allocator;

			dims = static_cast<int>(shape.Size());

			cstep = dims > 1 ? (size_t)shape[dims - 1LL] * shape[dims - 2LL] : shape[0];
			if (aligned) cstep = AlignSize(cstep * (depth >> DEPTH_SHIFT), MALLOC_ALIGN) / (depth >> DEPTH_SHIFT);

			if (Total() > 0)
			{
				size_t size = AlignSize(Total() * (depth >> DEPTH_SHIFT), 4);
				if (allocator)
					data = allocator->FastMalloc(size + sizeof(*ref_cnt));
				else
					data = FastMalloc(size + sizeof(*ref_cnt));
				ref_cnt = (int*)(((uchar*)data) + size);
				*ref_cnt = 1;
			}
		}


		void Tensor::Release()
		{
			if (ref_cnt && CHAOS_XADD(ref_cnt, -1) == 1)
			{
				if (allocator)
					allocator->FastFree(data);
				else
					FastFree(data);
			}

			Shape().Swap(shape);
			depth = U8;
			dims = 0;
			aligned = false;

			data = nullptr;
			ref_cnt = nullptr;
		}

		Tensor Tensor::Flatten() const
		{
			if (IsContinue())
			{
				return *this;
			}
			else
			{
				Tensor flattened = Tensor(shape, depth, /*aligned=*/false, allocator);
				auto num = Total() / cstep;
				for (int n = 0; n < num; n++)
				{
					const void* src = (unsigned char*)data + n * cstep * (depth >> DEPTH_SHIFT);
					void* dst = (unsigned char*)flattened.data + n * flattened.cstep * (flattened.depth >> DEPTH_SHIFT);
					memcpy(dst, src, flattened.cstep * (flattened.depth >> DEPTH_SHIFT));
				}
				return flattened;
			}
		}

		Tensor Tensor::Reshape(const Shape& new_shape)
		{
			size_t new_size = 1;
			for (auto n : new_shape)
			{
				new_size *= n;
			}
			CHECK_EQ(Size(), new_size) << "Must keep the number of elements same.";

			Tensor new_tensor = Tensor(new_shape, depth, aligned, allocator);

			if (!aligned)
			{
				//new_tensor = Tensor(new_shape, depth, data, aligned, allocator); //???ËÆºõ¡£¡£¡£ÓÐÎÊÌâ°¡¡£¡£
				memcpy(new_tensor.data, data, Size() * depth); 
			}
			else
			{
				// Flatten
				//new_tensor = Tensor(new_shape, depth, aligned, allocator);
				Tensor flattened = Flatten();

				// Copy to new shape
				auto new_num = new_tensor.Total() / new_tensor.cstep;
				auto csize = new_tensor.Size() / new_num;
				for (int n = 0; n < new_num; n++)
				{
					const void* src = (unsigned char*)flattened.data + n * csize * (flattened.depth >> DEPTH_SHIFT);
					void* dst = (unsigned char*)new_tensor.data + n * new_tensor.cstep * (new_tensor.depth >> DEPTH_SHIFT);
					memcpy(dst, src, csize * (new_tensor.depth >> DEPTH_SHIFT));
				}
			}

			return new_tensor;
		}

		size_t Tensor::Total() const
		{
			size_t total = cstep;
			for (int i = 0; i < dims - 2; i++) total *= shape[i];
			return total;
		}

		size_t Tensor::Size() const
		{
			size_t size = 1;
			for (auto val : shape) size *= val;
			return size;
		}

		bool Tensor::IsContinue() const
		{
			if (dims < 3) return true;
			return cstep == (size_t)shape[dims - 1LL] * shape[dims - 2LL];
		}

		inline std::ostream& operator<<(std::ostream& stream, const Tensor& tensor)
		{
			if (tensor.dims == 1)
			{
				stream << Mat(1, tensor.shape[0], Cast(tensor.depth), tensor.data) << std::endl;
			}
			else
			{
				int num = 1;
				for (int i = 0; i < tensor.dims - 2; i++)
				{
					num *= tensor.shape[i];
				}

				int h = tensor.shape[tensor.dims - 2LL];
				int w = tensor.shape[tensor.dims - 1LL];
				char* slice = (char*)tensor.data;
				for (int i = 0; i < num; i++)
				{
					stream << Mat(h, w, Cast(tensor.depth), slice) << std::endl;
					slice += tensor.cstep * (tensor.depth >> DEPTH_SHIFT);
				}
				//stream << Mat(h, w, Cast(tensor.depth), slice);
			}
			stream << "<Tensor " << tensor.shape << ", dtype=" << ToString(tensor.depth) << ">";
			return stream;
		}
	}
}