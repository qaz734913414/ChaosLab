#include <chaoscv.hpp>

#include <random>
#include <chrono>

DEFINE_STRING(data, "", "", "Data folder");
DEFINE_STRING(database, "", "", "Database for testing");
DEFINE_STRING(gallery, "", "", "Gallery Test");
DEFINE_STRING(genuine, "", "", "Genuine Test");
DEFINE_BOOL(shuffle, "", "Shuffle the index when CreateDB");
DEFINE_BOOL(version, "", "Show version");

DEFINE_STRING(symbol, "", "Deep Learning", "Symbol file for face feature model");
DEFINE_STRING(weight, "", "Deep Learning", "Weight file for face feature model");
DEFINE_STRING(layer_name, "fc1_output", "Deep Learning", "Output layer name");
DEFINE_INT(device_id, 0, "Deep Learning", "Device ID");
DEFINE_BOOL(use_gpu, "Deep Learning", "Use GPU if true");

DEFINE_STRING(mtcnn, "", "Face", "MTCNN model folder");
DEFINE_INT(height, 112, "Face", "Input height for face model");
DEFINE_INT(width, 112, "Face", "Input width for face model");
DEFINE_FLOAT(pad, 0, "Face", "Padding for aligner");


using namespace chaos;
using namespace chaos::face;
using namespace chaos::dnn;
using namespace chaos::test;

using Func = std::function<void()>;
using FuncMap = std::map<std::string, Func>;
static FuncMap func_map;

#define REGISTERFUNC(func)             \
namespace {                            \
  class Registerer_##func {            \
    public: /* NOLINT */               \
    Registerer_##func() {              \
      func_map[#func] = &func;         \
    }                                  \
  };                                   \
  Registerer_##func registerer_##func; \
}
static Func GetFunction(const std::string& name)
{
	if (func_map.count(name))
	{
		return func_map[name];
	}
	else
	{
		LOG(chaos::ERROR) << "Available actions:";
		for (FuncMap::iterator it = func_map.begin(); it != func_map.end(); ++it)
		{
			LOG(chaos::ERROR) << "\t" << it->first;
		}
		LOG(chaos::FATAL) << "Unknown action: " << name;
		return Func();  // not reachable, just to suppress old compiler warnings.
	}
}

void CreateDB()
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	auto gallery = DataWriter::Create(flag_gallery);
	auto genuine = DataWriter::Create(flag_genuine);

	for (auto& p : flag_data)
	{
		p = p == '/' ? '\\' : p;
	}
	if (flag_data.back() != '\\') flag_data.push_back('\\');

	FileList list;
	ProgressBar::Render("Searching");
	GetFileList(flag_data, list, "jpg|jpeg|png|bmp", ProgressBar::Update);
	ProgressBar::Halt();

	std::map<std::string, FileList> group;
	ProgressBar::Render("Tidying", list.size());
	for (auto file : list)
	{
		std::string name = file.Path.substr(flag_data.size());
		name.pop_back();
		group[name].push_back(file);
		ProgressBar::Update();
	}
	ProgressBar::Halt();

	ProgressBar::Render("Creating", group.size());
	int idx = 0;
	for (auto g : group)
	{
		if (flag_shuffle) std::shuffle(g.second.begin(), g.second.end(), std::default_random_engine(seed));

		Label label = (CLabel() << idx++);
		Sample sample = FileList{ g.second[0] };
		gallery->Put(sample, label);

		for (int i = 1; i < g.second.size(); i++)
		{
			Sample sample = FileList{ g.second[i] };
			genuine->Put(sample, label);
		}
		ProgressBar::Update();
	}
	ProgressBar::Halt();

	gallery->Close();
	genuine->Close();
}
REGISTERFUNC(CreateDB);

void Detect()
{
	Context ctx = Context(flag_use_gpu ? GPU : CPU, flag_device_id);
	auto detector = Detector::LoadMTCNN(flag_mtcnn, ctx);
	auto aligner = Aligner::CreateL5(Size(flag_width, flag_height), flag_pad);

	FileList list;
	ProgressBar::Render("Searching");
	GetFileList(flag_data, list, "jpg|jpeg|bmp|png|JPG|JPEG|PNG|BMP", ProgressBar::Update);
	ProgressBar::Halt();

	ProgressBar::Render("Detecting", list.size());
	for (auto file : list)
	{
		Mat image = cv::imread(file);
		Rect center(image.cols / 4.f, image.rows / 4.f, image.cols / 2.f, image.rows / 2.f);

		auto faces_info = detector->Detect(image);
		if (!faces_info.empty())
		{
			Sort(center, faces_info);

			Mat face = aligner->Align(image, faces_info[0].points);
			cv::imwrite(file, face);
		}
		else
		{
			LOG(WARNING) << "Can not detect face from file " << file;
			Delete(file);
		}
		ProgressBar::Update();
	}
	ProgressBar::Halt();
}
REGISTERFUNC(Detect);

void Test()
{
	Context ctx = Context(flag_use_gpu ? GPU : CPU, flag_device_id);
	auto net = Net::Load({flag_symbol, flag_weight}, ctx);
	net->BindExecutor({ {"data", {1, 3, flag_height, flag_width}} });

	auto engine = ITest::Create(flag_database);
	engine->Gallery = DataLoader::Load(flag_gallery);
	engine->Genuine = DataLoader::Load(flag_genuine);

	engine->Forward = [=](const Mat& image)->Mat {
		Mat data;
		image.convertTo(data, CV_32F);

		Tensor feat;
		net->SetLayerData("data", Tensor::Unroll({ data }, true));
		net->Forward();
		net->GetLayerData(flag_layer_name, feat);

		return Mat(feat.dims, feat.shape.data(), CV_32F, feat.data).clone();
	};

	engine->Run();
	engine->Report();
	engine->Save();
	engine->Close();
}
REGISTERFUNC(Test);
 
int main(int argc, char** argv)
{
	SetUsageMessage(
		"\n"
		"FaceBench commond <args>\n"
		"commond:\n"
		"    Test          To test the performance\n"
		"                  Use gallery and genuine to test identify performance\n"
		"    Detect        To detect the face\n"
		"                  Use MTCNN to detect face\n"
		"    CreateDB      To create database\n"
		"                  This is just an example"
	);

	ParseCommondLineFlags(&argc, &argv);
	InitLogging(argv[0]);

	

	if (argc == 2 && !flag_version)
	{
		GetFunction(argv[1])();
	}
	else if (flag_version)
	{
		std::cout << "Face Benchmark v4" << std::endl;
		std::cout << GetVersionInfo() << std::endl;
		std::cout << GetMXVI() << std::endl;
	}
	else
	{
		ShowUsageMessage("benchmark.cpp");
	}

	return 0;
}