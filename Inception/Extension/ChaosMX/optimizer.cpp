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

			void MergeBatchNorm() final
			{
				for (size_t i = 0; i < symbols.size(); i++)
				{
					if ("BatchNorm" == symbols[i].op.op_name)
					{
					}
				}
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

					//auto name = Split(names[i], ":")[1];
					weights[names[i]] = data;

					CHECK_EQ(0, MXNDArrayFree(handles[i])) << MXGetLastError();
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
						for (const auto& attr : sym.attrs)
						{
							config_keys.push_back(attr.first.c_str());
							config_vals.push_back(attr.second.c_str());
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
				for (auto& h : handles)
				{
					CHECK_EQ(0, MXSymbolFree(h)) << MXGetLastError();
				}
			}
			void SaveWeight(const std::string& file)
			{
				std::vector<NDArrayHandle> handles;
				std::vector<const char*> keys;
				for (const auto& w : weights)
				{
					NDArrayHandle handle;

					std::vector<mx_uint> shape = w.second.shape;
					CHECK_EQ(0, MXNDArrayCreate(shape.data(), shape.size(), 1, 0, false, &handle)) << MXGetLastError();

					void* pdata = nullptr;
					CHECK_EQ(0, MXNDArrayGetData(handle, &pdata)) << MXGetLastError();

					memcpy(pdata, w.second.data, w.second.Size());

					handles.push_back(handle);
					keys.push_back(w.first.c_str());
				}

				CHECK_EQ(0, MXNDArraySave(file.c_str(), handles.size(), handles.data(), keys.data())) << MXGetLastError();

				// Release
				for (auto& h : handles)
				{
					CHECK_EQ(0, MXNDArrayFree(h)) << MXGetLastError();
				}
			}

			std::map<std::string, Tensor> weights;
			std::vector<Symbol> symbols;

		};

		Ptr<Optimizer> Optimizer::LoadMxNet(const Model& model)
		{
			return Ptr<Optimizer>(new MxOp(model));
		}
	}
}