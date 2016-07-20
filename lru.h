#pragma once

// Least-recently used cache for storing lidar scans
// because we can't keep all 5000 in memory.
// Each scan has 130,000 points, each taking up 16 bytes
// not counting the duplication and overhead in the kd tree
struct ScanData {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans;
    std::vector<pcl::KdTreeFLANN<pcl::PointXYZ>> trees;
    int _frame;
    ScanData() {}
    ScanData(const std::string dataset, const int frame) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
        loadPoints(cloud, dataset, frame);
        segmentPoints(cloud, scans);
        trees.resize(scans.size());
        for(int i=0; i<scans.size(); i++) {
            trees[i].setInputCloud(scans[i]);
        }
        _frame = frame;
        /*
        std::cerr << "created scandata: " << dataset 
            << ", " << frame 
            << ": " << scans.size()
            << std::endl;
            */
    }
};

class ScansLRU {
    private:
    const int size = 50;
    std::list<ScanData*> times;
    std::unordered_map<int, decltype(times)::iterator> exists;
    public:
    ScanData* get(const std::string dataset,
            const int frame
            ) {
        // retrieves from scan if possible,
        // loads data from disk otherwise
        if(exists.count(frame)) {
            auto it = exists[frame];
            ScanData *sd = *it;
            times.erase(it);
            times.push_front(sd);
            return sd;
        } else {
            ScanData *sd = new ScanData(dataset, frame);
            times.push_front(sd);
            exists[frame] = times.begin();
            if(times.size() > size) {
                auto sd = times.back();
                exists.erase(sd->_frame);
                delete sd;
                times.pop_back();
            }
            return sd;
        }
    }
};
