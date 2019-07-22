#include "base.hpp"
#include "symbol.hpp"

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
				CHECK(model.from_file) << "Just support to load from file";

				LoadSymbol(model.symbol);
				LoadWeight(model.weight);

			}

			~MxOp()
			{
				//CHECK_EQ(0, MXSymbolFree(symbol)) << MXGetLastError();
			}

			void Export(const std::string& name) final
			{
				SaveSymbol(name + ".json");
				SaveWeight(name + ".params");
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

				Json symbol_json = Shrink(json);

				auto nodes = symbol_json["nodes"];
				size_t cnt = nodes.Data.size();
				for (size_t i = 0; i < cnt; i++)
				{
					symbols.push_back(nodes[i]);
				}
			}

			void SaveSymbol(const std::string& file)
			{
				std::vector<SymbolHandle> handles;
				for (auto sym : symbols)
				{
					SymbolHandle handle;

					if (sym.op.op_name == "null")
					{
						CHECK_EQ(0, MXSymbolCreateVariable(sym.name.c_str(), &handle)) << MXGetLastError();
						for (auto attr : sym.attrs)
						{
							CHECK_EQ(0, MXSymbolSetAttr(handle, attr.first.c_str(), attr.second.c_str())) << MXGetLastError();
						}
					}
					else
					{
						std::vector<const char*> config_keys;
						std::vector<const char*> config_vals;
						for (auto& attr : sym.attrs)
						{
							config_keys.push_back(attr.first.data());
							config_vals.push_back(attr.second.data());
						}
						CHECK_EQ(0, MXSymbolCreateAtomicSymbol(sym.op.creator, (mx_uint)sym.attrs.size(), config_keys.data(), config_vals.data(), &handle) ) << MXGetLastError();

						std::vector<const char*> inputs_keys;
						std::vector<SymbolHandle> inputs_vals;
						for (size_t i = 0; i < sym.inputs.size(); i++)
						{
							inputs_keys.push_back(sym.op.op_info.arg_names[i]);
							inputs_vals.push_back(handles[sym.inputs[i].node_id]);
						}
						CHECK_EQ(0, MXSymbolCompose(handle, sym.name.c_str(), (mx_uint)sym.inputs.size(), inputs_keys.data(), inputs_vals.data())) << MXGetLastError();
					}
					handles.push_back(handle);
				}

				CHECK_EQ(0, MXSymbolSaveToFile(handles.back(), file.c_str())) << MXGetLastError();

				// Release
				for (auto h : handles)
				{
					CHECK_EQ(0, MXSymbolFree(h)) << MXGetLastError();
				}
			}

			void SaveWeight(const std::string& file)
			{

			}

			std::map<std::string, Tensor> weights;
			std::vector<Symbol> symbols;

		};

		Ptr<Optimizer> Optimizer::LoadMX(const Model& model)
		{
			return Ptr<Optimizer>(new MxOp(model));
		}
	}
}