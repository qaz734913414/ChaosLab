#include "utils/numpy.hpp"
#include "utils/json.hpp"

#include <fstream>

namespace chaos
{
	std::string Cast(const dnn::Depth& depth)
	{
		switch (depth)
		{
		case dnn::F32:
			return "<f4";
		case dnn::U8:
			return "<i4";
		default:
			LOG(FATAL) << "Now just support F32 and U8";
			return "";
		}
	}

	dnn::Depth Cast(const std::string& descr)
	{
		switch (Hash(descr.c_str()))
		{
		case "'<f4'"_hash:
			return dnn::F32;
		case "'<i4'"_hash:
			return dnn::U8;
		default:
			LOG(FATAL) << "Now do not support " << descr;
			return dnn::UNK;
		}
	}

	Numpy::Numpy(const File& file) : file(file) { CHECK_EQ("npy", file.Type); }

	void Numpy::CreateHead(const dnn::Shape& shape, const dnn::Depth& depth)
	{
		std::string head = cv::format("%cNUMPY%c%cv%c{'descr': '%s', 'fortran_order': False, 'shape': %s, }", 0x93, 0x01, 0x00, 0x00, Cast(depth).c_str(), shape.ToString().c_str());
		int remain = 127 - (int)head.size();
		for (int i = 0; i < remain; i++)
		{
			head.push_back(' ');
		}
		head.push_back(0x0A);

		std::ofstream fs(file, std::ios::binary);
		fs.write(head.c_str(), head.size());
		fs.close();
	}

	void Numpy::Add(const dnn::Tensor& tensor)
	{
		CHECK_EQ(false, tensor.aligned);

		std::ofstream fs(file, std::ios::binary | std::ios::app);
		fs.write((char*)tensor.data, tensor.Total() * tensor.depth);
		fs.close();
	}

	dnn::Tensor Numpy::Load(const File& file)
	{
		std::ifstream fs(file, std::ios::binary);
		std::string head(128, 0x00);

		fs.read(head.data(), 128);

		Json info = Shrink(head.substr(10, 117));
		auto shape = Split(info.Data["shape"], "\\(|,|\\)");

		std::vector<int> size;
		for (int i = 1; i < shape.size(); i++)
		{
			size.push_back(std::atoi(shape[i].c_str()));
		}
		dnn::Depth depth = Cast(info.Data["descr"]);

		dnn::Tensor tensor = dnn::Tensor(size, depth);
		fs.read((char*)tensor.data, depth * tensor.Size());

		fs.close();
		return tensor;
	}
}