#include "dnn/group.hpp"

namespace chaos
{
	namespace dnn
	{
		GroupNet::Load::Load(const Model& model, const Context& ctx) : model(model), ctx(ctx) {}
		GroupNet::Load& GroupNet::Load::As(const std::string& _name)
		{
			name = _name;
			return *this;
		}
		void GroupNet::Load::InTo(GroupNet& nets)
		{
			nets.Add(name, model, ctx);
		}

		GroupNet::GroupNet() {}
		GroupNet& GroupNet::Add(const std::string& name, const Model& model, const Context& ctx)
		{
			nets[name] = Net::Load(model, ctx);
			return *this;
		}

		GroupNet& GroupNet::Forward(const std::string& name)
		{
			forward_func[name]();
			return *this;
		}
		GroupNet& GroupNet::SetForward(const std::string& name, const std::function<void()>& func)
		{
			forward_func[name] = func;
			return *this;
		}
		Ptr<Net>& GroupNet::operator[](const std::string& name)
		{
			return nets[name];
		}
	} // namespace dnn
} // namespace chaos