#include "dnn/optimizer.hpp"
#include "symbol.hpp"

#pragma warning (push, 0)
#include <mxnet/c_api.h>
#pragma warning (pop)

#include <stack>

namespace chaos
{
	namespace dnn
	{
		class MxOp : public Optimizer
		{
		public:
			MxOp(const Model& model)
			{
				mx_uint num_symbol_creators = 0;
				AtomicSymbolCreator* creators = nullptr;
				CHECK_EQ(0, MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &creators)) << MXGetLastError();
				for (mx_uint i = 0; i < num_symbol_creators; i++)
				{
					const char* name;
					const char* description;
					mx_uint num_args;
					const char** arg_names;
					const char** arg_type_infos;
					const char** arg_descriptions;
					const char* key_var_num_args;
					CHECK_EQ(0, MXSymbolGetAtomicSymbolInfo(creators[i], &name, &description,
						&num_args, &arg_names, &arg_type_infos,
						&arg_descriptions, &key_var_num_args)) << MXGetLastError();
					symbol_creators[name] = creators[i];

					for (mx_uint i = 0; i < num_args; i++)
					{
						symbol_args[name].push_back(arg_names[i]);
					}
				}

				CHECK(model.from_file) << "Just support to load from file";

				LoadSymbol(model.symbol);
				LoadWeight(model.weight);

				//AtomicSymbolHandle atomic;
				//SymbolHandle handle;
				//CHECK_EQ(0, MXSymbolCreateVariable("data", &handle)) << MXGetLastError();
				//SymbolHandle h2;
				//MXSymbolCreateAtomicSymbol(symbol_creators["Convolution"], 0, nullptr, nullptr, &h2);
				//MXSymbolCompose(handle, "conv1", 0, nullptr, &h2);
				//CHECK_EQ(0, MXSymbolSaveToFile(handle, "test.json")) << MXGetLastError();
			}

			~MxOp()
			{
				//CHECK_EQ(0, MXSymbolFree(symbol)) << MXGetLastError();
			}

			void Export(const std::string& name) final
			{

			}

		private:
			void LoadWeight(const std::string& file) 
			{
				NDArrayHandle* handles = nullptr;
				mx_uint cnt, ncnt;
				const char** names = nullptr;
				CHECK_EQ(0, MXNDArrayLoad(file.c_str(), &cnt, &handles, &ncnt, &names)) << MXGetLastError();

				for (mx_uint i = 0; i < cnt; i++)
				{
					mx_uint dims;
					const mx_uint* pdata;
					CHECK_EQ(0, MXNDArrayGetShape(handles[i], &dims, &pdata)) << MXGetLastError();
					std::vector<mx_uint> shape;
					for (mx_uint j = 0; j < dims; j++)
						shape.push_back(pdata[j]);

					Tensor data = Tensor(shape, F32);

					void* buff = nullptr;
					CHECK_EQ(0, MXNDArrayGetData(handles[i], &buff)) << MXGetLastError();

					memcpy(data.data, buff, data.Size());

					auto name = Split(names[i], ":")[1];
					weights[name] = data;
				}
			}
			
			void LoadSymbol(const std::string& file)
			{

				std::fstream fs(file, std::ios::in | std::ios::binary);
				CHECK(fs.good()) << "Can not load weight file.";

				fs.seekg(0, std::ios::end);
				size_t size = fs.tellg();
				fs.seekg(0, std::ios::beg);

				std::string json(size, 0);

				fs.read((char*)json.data(), size);
				fs.close();

				symbols = Load(Shrink(json));

				

				// To Save another Symbol
				std::vector<SymbolHandle> handles;
				for (const auto& ss : symbols)
				{
					SymbolHandle handle;
					if (ss.op == "null")
					{
						CHECK_EQ(0, MXSymbolCreateVariable(ss.name.c_str(), &handle)) << MXGetLastError();
						for (auto attr : ss.attrs)
						{
							CHECK_EQ(0, MXSymbolSetAttr(handle, attr.first.c_str(), attr.second.c_str())) << MXGetLastError();
						}
					}
					else
					{
						std::vector<const char*> key;
						std::vector<const char*> vals;
						int i = 0;
						for (const auto& attr : ss.attrs)
						{
							key.push_back(attr.first.c_str());
							vals.push_back(attr.second.c_str());
							i++;
						}

						CHECK_EQ(0, MXSymbolCreateAtomicSymbol(symbol_creators[ss.op], ss.attrs.size(), key.data(), vals.data(), &handle)) << MXGetLastError();
					}

					if (!ss.inputs.empty())
					{
						std::vector<SymbolHandle> inputs;
						std::vector<const char*> keys;
						int i = 0;
						for (const auto& in : ss.inputs)
						{
							inputs.push_back(handles[in.node_id]);

							std::string op = ss.op;
							keys.push_back(symbol_args[op][i++].c_str());
							//std::string key = symbols[in.node_id].name;
							//key = key.find(ss.name) < key.size() ? key.substr(ss.name.size() + 1) : "data";
							//keys.push_back(new char[key.size() + 1]());
							//memcpy((char*)keys.back(), key.data(), key.size());
						}
						CHECK_EQ(0, MXSymbolCompose(handle, ss.name.c_str(), keys.size(), keys.data(), inputs.data())) << MXGetLastError();
					}
					handles.push_back(handle);
				}
				MXSymbolSaveToFile(handles.back(), "test.json");
			}

			std::map<std::string, Tensor> weights;
			//SymbolHandle symbol;
			SymbolList symbols;

			std::map<std::string, AtomicSymbolCreator> symbol_creators;
			std::map<std::string, std::vector<std::string>> symbol_args;

		};

		Ptr<Optimizer> Optimizer::LoadMX(const Model& model)
		{
			return Ptr<Optimizer>(new MxOp(model));
		}
	}
}