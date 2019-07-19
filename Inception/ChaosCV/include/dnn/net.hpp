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

		// Pre-declaration
		class Framework;
		/// <summary>Net just for inference</summary>
		class CHAOS_API Net
		{
		public:
			virtual ~Net();

			/// <summary>Bind the executor</summary>
			/// <param name="inputs">Inputs info</param>
			virtual void BindExecutor(const std::vector<DataLayer>& inputs) = 0;
			/// <summary>Forward the newtork</summary>
			virtual void Forward() = 0;
			/// <summary>Set the layer data</summary>
			/// <param name="name">Layer name</param>
			/// <param name="data">Tensor data</param>
			virtual void SetLayerData(const std::string& name, const Tensor& data) = 0;
			/// <summary>Get the layer data</summary>
			/// <param name="name">Layer name</param>
			/// <param name="data">Tensor data</param>
			virtual void GetLayerData(const std::string& name, Tensor& data) = 0;
			/// <summary>Reshape the network if supported</summary>
			/// <param name="inputs">New inputs info</param>
			virtual void Reshape(const std::vector<DataLayer>& new_inputs) = 0;

			/// <summary>Get the framework</summary>
			virtual dnn::Framework& GetFramework() = 0;
			__declspec(property(get = GetFramework)) dnn::Framework& Framework;

			static Ptr<Net> Load(const dnn::Model& model, const Context& ctx = Context());
		};

		CHAOS_API Ptr<Net> LoadMxNet(const Model& model, const Context& ctx = Context());
		//CHAOS_API Ptr<Net> LoadVINO(const Model& model, const Context& ctx = Context());


		class CHAOS_API Framework
		{
		public:
			using LoadFunction = std::function<Ptr<Net>(const Model&, const Context&)>;

			Framework();
			Framework(const std::string& name);
			Framework& With(const std::string& sym_type, const std::string& wgt_type, const LoadFunction& func);

			std::string name;
			std::string symbol_type;
			std::string weight_type;
			LoadFunction load_func;
		};

	} // namespace dnn
} // namespace chaos