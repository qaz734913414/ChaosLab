#include "core/core.hpp"
#include "core/version.hpp"

#include <regex>

namespace chaos
{
	bool ProgressBar::running = false;
	bool ProgressBar::stopped = true;
	char* ProgressBar::progress = nullptr;
	size_t ProgressBar::current = 0;
	std::mutex ProgressBar::mtx;
	std::condition_variable ProgressBar::halted;

	void ProgressBar::Render(const std::string& message, size_t total, int len)
	{
		if (progress)
		{
			delete[] progress;
			progress = nullptr;
		}

		current = 0;
		running = true;
		stopped = false;
		std::thread pbar(Refresh, message, total, len);
		pbar.detach();
	}

	void ProgressBar::Update(int step)
	{
		current += step;
	}

	void ProgressBar::Halt()
	{
		running = false;
		std::unique_lock<std::mutex> lock(mtx);
		halted.wait(lock, [] { return stopped; });
		lock.unlock();
	}

	void ProgressBar::Refresh(const std::string& message, size_t total, const int len)
	{
		std::unique_lock<std::mutex> lock(mtx);

		const char sign[] = { '>', ' ' };

		progress = new char[len + 1LL]();
		memset(progress, ' ', len);

		// Start status
		std::string pbar = message + cv::format(" |%s|", progress) +
			(total > 0 ? "   0.00%% 0/" + std::to_string(total) : " 0");
		std::cout << pbar;
		for (auto c : pbar) std::cout << "\b";

		double during;
		int hour, min, sec;
		int idx = 0;
		int64 start = cv::getTickCount();
		int64 pre = start;
		while (running)
		{
			int64 now = cv::getTickCount();
			if (0.2 - (now - pre) / cv::getTickFrequency() <= 1e-5)
			{
				pre = now;
				during = (now - start) / cv::getTickFrequency();
				hour = (int)(during / 3600);
				min = (int)((during - hour * 3600.) / 60);
				sec = (int)(during - hour * 3600. - min * 60.);

				if (total > 0)
				{
					double percent = (current + 0.) / total;
					idx = std::min(len, (int)(percent * len));
					memset(progress, sign[0], idx);
					pbar = message + cv::format(" |%s| %6.2lf%% %lld/%lld [%03d:%02d:%02d]", progress, percent * 100., current, total, hour, min, sec);
				}
				else
				{
					idx++;
					int i = idx / len % 2;
					int j = std::min(len, idx % len + 1);
					memset(progress, sign[i], j);
					pbar = message + cv::format(" |%s| %lld [%03d:%02d:%02d]", progress, current, hour, min, sec);
				}
				std::cout << pbar;
				for (auto c : pbar) std::cout << "\b";
			}
		}

		during = (cv::getTickCount() - start) / cv::getTickFrequency();
		hour = (int)(during / 3600);
		min = (int)((during - hour * 3600.) / 60);
		sec = (int)(during - hour * 3600. - min * 60.);

		double percent = (current + 0.) / total;
		idx = total > 0 ? std::min(len, (int)(percent * len)) : len;
		memset(progress, '>', idx);

		pbar = message + cv::format(" |%s|", progress);
		pbar += (total > 0 ? cv::format(" %6.2lf%% %lld/%lld", percent * 100, current, total) : cv::format(" %lld", current));
		pbar += cv::format(" [%03d:%02d:%02d]", hour, min, sec);

		std::cout << pbar << std::endl;

		delete[] progress;
		progress = nullptr;

		stopped = true;

		lock.unlock();
		halted.notify_one();
	}

	std::vector<std::string> Split(const std::string& data, const std::string& delimiter)
	{
		std::regex regex{ delimiter };
		return std::vector<std::string> {
			std::sregex_token_iterator(data.begin(), data.end(), regex, -1),
				std::sregex_token_iterator()
		};
	}
	std::string Shrink(const std::string& data)
	{
		std::string shrinked;
		for (auto c : data)
		{
			if (c != ' ' && c != 10) shrinked.push_back(c);
		}
		return shrinked;
	}

} // namespace chaos

#include <Windows.h>

std::string chaos::GetVersionInfo()
{
	std::stringstream ss;
	ss << "ChaosCV " << RELEASE_VER_STR << " [MSC v." << _MSC_VER << " " << __T(_MSC_PLATFORM_TARGET) << "bit]";
	return ss.str();
}

void chaos::SetConsoleTextColor(unsigned short color)
{
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color | FOREGROUND_INTENSITY);
}