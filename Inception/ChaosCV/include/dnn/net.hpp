#pragma once

#include "dnn/tensor.hpp"
#include "dnn/layers/data_layer.hpp"

namespace chaos
{
	namespace dnn
	{
		enum DeviceType
		{
			CPU,
			GPU,
		};

		class CHAOS_API Context
		{
		public:
			Context();
			Context(const DeviceType& type, int id = 0);

			DeviceType type = CPU;
			int id = 0;
		};

		class CHAOS_API Model
		{
		public:
			Model();
			Model(const std::string& weight);
			Model(const std::string& symbol, const std::string& weight);

			std::string symbol;
			std::string weight;

			bool from_file = true;
		};

		/// Refer to MxNet and NCNN
		class CHAOS_API Net
		{
		public:
			virtual ~Net();

			// 正常前向传播

			virtual void BindExecutor(const std::vector<DataLayer>& inputs) = 0;

			/// <summary>Forward the newtork</summary>
			virtual void Forward() = 0;

			virtual void SetLayerData(const std::string& name, const Tensor& data) = 0;
			virtual void GetLayerData(const std::string& name, Tensor& data) = 0;

			virtual void Reshape(const std::vector<DataLayer>& inputs) = 0;

			// 优化？还是另外读取模型？
		};

		CHAOS_API Ptr<Net> LoadMxNet(const Model& model, const Context& ctx = Context());
	} // namespace dnn
} // namespace chaos