#pragma once

#include "dnn/net.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API GroupNet
		{
		public:
			class CHAOS_API Load
			{
			public:
				Load(const Model& model, const Context& ctx);
				Load& As(const std::string& name);
				void InTo(GroupNet& nets);

			private:
				Model model;
				Context ctx;
				std::string name;
			};

			GroupNet();

			GroupNet& Add(const std::string& name, const Model& model, const Context& ctx = Context());
			GroupNet& Forward(const std::string& name);
			GroupNet& SetForward(const std::string& name, const std::function<void()>& func);

			Ptr<Net>& operator[](const std::string& name);
		private:
			std::map<std::string, std::function<void()>> forward_func; // <name, func>
			std::map<std::string, Ptr<Net>> nets;
		};
	}
}