#include "dnn/net.hpp"
#include "dnn/reg.hpp"

#include "base.hpp"

namespace chaos
{
	namespace dnn
	{
		class Predictor : public Net
		{
		public:
			Predictor(const Model& model, const Context& ctx)
			{
				switch (ctx.type)
				{
				case GPU:
					dev_type = 2;
					break;
				case CPU:
					dev_type = 1;
					break;
				default:
					LOG(WARNING) << "MxNet now do not support device type " << ctx.type << ", use CPU mode for default.";
					dev_type = 1;
					break;
				}
				dev_id = ctx.id;

				if (model.from_file)
				{
					LoadSymbol(model.symbol);
					LoadWeight(model.weight);
				}
				else
				{
					symbol = model.symbol;
					weight = model.weight;
					GetOutputInfo();
				}
			}

			~Predictor()
			{
				CHECK_EQ(0, MXPredFree(predictor)) << MXGetLastError();
			}

			void BindExecutor(const std::vector<DataLayer>& inputs) final
			{
				for (auto layer : inputs)
				{
					shapes[layer.name] = layer.shape;
				}

				int size = (int)inputs.size();

				std::vector<const char*> input_keys;
				std::vector<mx_uint> indptr;
				std::vector<mx_uint> shape_data;

				GetInputsInfo(inputs, input_keys, indptr, shape_data);

				CHECK_EQ(0, MXPredCreate(symbol.data(), weight.data(), (int)weight.size(),
					dev_type, dev_id, size, input_keys.data(),
					indptr.data(), shape_data.data(), &predictor)) << MXGetLastError();
			}

			/// <summary>Forward the newtork</summary>
			void Forward() final
			{
				CHECK_EQ(0, MXPredForward(predictor)) << MXGetLastError();
			}

			void SetLayerData(const std::string& name, const Tensor& data) final
			{
				CHECK_EQ(F32, data.depth);
				CHECK_EQ(shapes[name], data.shape);

				CHECK_EQ(0, MXPredSetInput(predictor, name.data(), (const float*)data.data, (mx_uint)data.Size())) << MXGetLastError();
			}
			void GetLayerData(const std::string& name, Tensor& data) final
			{
				CHECK(output_idx.find(name) != output_idx.end());

				mx_uint* shape;
				mx_uint dims;
				CHECK_EQ(0, MXPredGetOutputShape(predictor, output_idx[name], &shape, &dims)) << MXGetLastError();

				std::vector<int> size(dims);
				for (mx_uint i = 0; i < dims; i++)
				{
					size[i] = shape[i];
				}
				data = Tensor(size, F32);

				CHECK_EQ(0, MXPredGetOutput(predictor, output_idx[name], (float*)data.data, (mx_uint)data.Size())) << MXGetLastError();
			}

			void Reshape(const std::vector<DataLayer>& new_inputs) final
			{
				for (auto layer : new_inputs)
				{
					CHECK_NE(shapes.end(), shapes.find(layer.name));
					shapes[layer.name] = layer.shape;
				}

				int size = (int)new_inputs.size();

				std::vector<const char*> input_keys;
				std::vector<mx_uint> indptr;
				std::vector<mx_uint> shape_data;

				GetInputsInfo(new_inputs, input_keys, indptr, shape_data);

				PredictorHandle new_predictor;
				CHECK_EQ(0, MXPredReshape(size, input_keys.data(),
					indptr.data(), shape_data.data(), predictor, &new_predictor)) << MXGetLastError();

				CHECK_EQ(0, MXPredFree(predictor)) << MXGetLastError();
				predictor = new_predictor;
			}

			dnn::Framework& GetFramework() final
			{
				return Registered::Have("MxNet");
			}

		private:
			void LoadWeight(const std::string& file)
			{
				std::fstream fs(file, std::ios::in | std::ios::binary);
				CHECK(fs.good()) << "Can not load weight file.";

				fs.seekg(0, std::ios::end);
				size_t size = fs.tellg();
				fs.seekg(0, std::ios::beg);

				weight.resize(size);

				fs.read((char*)weight.data(), size);
				fs.close();
			}

			void LoadSymbol(const std::string& file)
			{
				SymbolHandle handle;
				CHECK_EQ(0, MXSymbolCreateFromFile(file.c_str(), &handle)) << MXGetLastError();

				// Save output names and idx
				mx_uint num = 0;
				const char** names;
				CHECK_EQ(0, MXSymbolListOutputs(handle, &num, &names)) << MXGetLastError();

				for (mx_uint i = 0; i < num; i++)
				{
					output_idx[names[i]] = i;
				}
				
				const char* buff;
				CHECK_EQ(0, MXSymbolSaveToJSON(handle, &buff)) << MXGetLastError();
				symbol = std::string(buff);

				CHECK_EQ(0, MXSymbolFree(handle)) << std::endl << MXGetLastError();
			}

			void GetOutputInfo()
			{
				SymbolHandle handle;
				CHECK_EQ(0, MXSymbolCreateFromJSON(symbol.c_str(), &handle)) << MXGetLastError();

				mx_uint num = 0;
				const char** names;
				CHECK_EQ(0, MXSymbolListOutputs(handle, &num, &names)) << MXGetLastError();

				for (mx_uint i = 0; i < num; i++)
				{
					output_idx[names[i]] = i;
				}

				CHECK_EQ(0, MXSymbolFree(handle)) << MXGetLastError();
			}

			void GetInputsInfo(const std::vector<DataLayer>& inputs,
				std::vector<const char*>& input_keys, std::vector<mx_uint>& indptr, std::vector<mx_uint>& shape_data)
			{
				indptr.push_back(0);

				for (const auto& layer : inputs)
				{
					input_keys.push_back(layer.name.data());

					mx_uint ind = indptr.back() + static_cast<mx_uint>(layer.shape.Size());
					indptr.push_back(ind);

					shape_data.insert(shape_data.end(), layer.shape.begin(), layer.shape.end());
				}
			}

			std::string weight;
			std::string symbol;

			int dev_type;
			int dev_id;

			std::map<std::string, int> output_idx;

			PredictorHandle predictor = nullptr;

			std::map<std::string, Shape> shapes; // Inputs shapes
		};

		Ptr<Net> LoadMxNet(const Model& model, const Context& ctx)
		{
			return Ptr<Net>(new Predictor(model, ctx));
		}


	}
}