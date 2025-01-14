#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// Structure to store each item's information
struct Item {
    std::string sta_code;
    std::string sku_code;
    double length;
    double width;
    double height;
    int qty;

    friend bool operator == (Item a,Item b) {
        return a.sku_code == b.sku_code;
    }
};

// Function to split a string based on a delimiter
std::vector<std::string> split( std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(s);
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool cmp(Item x,Item y) {
    return (x.height * x.width * x.length > y.height * y.width * y.length);
}

int main() {
    std::string filename = "../input.csv"; // Name of the CSV file

    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return 1;
    }

    std::string line;
    std::vector<Item> items; // Vector to store all items


    // Read the first line (header)
    if (!std::getline(file, line)) {
        std::cerr << "CSV file is empty or cannot be read." << std::endl;
        return 1;
    }

    // Read each subsequent line and parse the data
    while (std::getline(file, line)) {
        // Split the line based on tab delimiter ('\t')
        std::vector<std::string> fields = split(line, ','); // Change to ',' if comma-separated

        // Check if the number of fields is correct
        if (fields.size() != 6) {
            std::cerr << "Invalid CSV line (expected 6 fields): " << line << std::endl;
            continue;
        }

        // Create an Item object and populate it
        Item item;
        item.sta_code = fields[0];
        item.sku_code = fields[1];

        try {
            item.length = std::stod(fields[2]);
            item.width = std::stod(fields[3]);
            item.height = std::stod(fields[4]);
            item.qty = std::stoi(fields[5]);
        } catch (std::invalid_argument& e) {
            std::cerr << "Invalid numerical value in line: " << line << std::endl;
            continue;
        }

        // Add the item to the vector
        items.push_back(item);
    }

    file.close(); // Close the file

    double a[9],b[9],c[9];
    std::cin >> a[1] >> b[1] >> c[1];
    std::cin >> a[2] >> b[2] >> c[2];
    std::cin >> a[3] >> b[3] >> c[3];
    std::cin >> a[4] >> b[4] >> c[4];
    std::cin >> a[5] >> b[5] >> c[5];
    std::cin >> a[6] >> b[6] >> c[6];
    std::cin >> a[7] >> b[7] >> c[7];
    std::cin >> a[8] >> b[8] >> c[8];

    std::vector<Item> items_OK1;

    for (auto& item : items) {
        if (item.length < a[1] && item.width < b[1] && item.height < c[1]) {
            items_OK1.push_back(item);
        }
    }

    std::sort(items_OK1.begin(),items_OK1.end(),cmp);

    double sum1 = 0;

    int i1 = 0;

    std::cout << "Read " << items_OK1.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK1) {
       std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum1 += item.length * item.width * item.height * item.qty;
        i1 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i1 > 1000) {
            break;
        }
    }

    std::cout << sum1;

    // =======

    std::vector<Item> items_OK2;

    for (auto& item : items) {
        if (item.length < a[2] && item.width < b[2] && item.height < c[2]) {
            items_OK2.push_back(item);
        }
    }

    std::sort(items_OK2.begin(),items_OK2.end(),cmp);

    double sum2 = 0;

    int i2 = 0;

    std::cout << "Read " << items_OK2.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK2) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum2 += item.length * item.width * item.height * item.qty;
        i2 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i2 > 1000) {
            break;
        }
    }

    std::cout << sum2;

    std::vector<Item> items_OK3;

    for (auto& item : items) {
        if (item.length < a[3] && item.width < b[3] && item.height < c[3]) {
            items_OK3.push_back(item);
        }
    }

    std::sort(items_OK3.begin(),items_OK3.end(),cmp);

    double sum3 = 0;

    int i3 = 0;

    std::cout << "Read " << items_OK3.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK3) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum3 += item.length * item.width * item.height * item.qty;
        i3 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i3 > 1000) {
            break;
        }
    }

    std::cout << sum3;

    std::vector<Item> items_OK4;

    for (auto& item : items) {
        if (item.length < a[4] && item.width < b[4] && item.height < c[4]) {
            items_OK4.push_back(item);
        }
    }

    std::sort(items_OK4.begin(),items_OK4.end(),cmp);

    double sum4 = 0;

    int i4 = 0;

    std::cout << "Read " << items_OK4.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK4) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum4 += item.length * item.width * item.height * item.qty;
        i4 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i4 > 1000) {
            break;
        }
    }

    std::cout << sum4;

    std::vector<Item> items_OK5;

    for (auto& item : items) {
        if (item.length < a[5] && item.width < b[5] && item.height < c[5]) {
            items_OK5.push_back(item);
        }
    }

    std::sort(items_OK5.begin(),items_OK5.end(),cmp);

    double sum5 = 0;

    int i5 = 0;

    std::cout << "Read " << items_OK5.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK5) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum5 += item.length * item.width * item.height * item.qty;
        i5 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i5 > 1000) {
            break;
        }
    }

    std::cout << sum5;

    std::vector<Item> items_OK6;

    for (auto& item : items) {
        if (item.length < a[6] && item.width < b[6] && item.height < c[6]) {
            items_OK6.push_back(item);
        }
    }

    std::sort(items_OK6.begin(),items_OK6.end(),cmp);

    double sum6 = 0;

    int i6 = 0;

    std::cout << "Read " << items_OK6.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK6) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum6 += item.length * item.width * item.height * item.qty;
        i6 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i6 > 1000) {
            break;
        }
    }

    std::cout << sum6;

    std::vector<Item> items_OK7;

    for (auto& item : items) {
        if (item.length < a[7] && item.width < b[7] && item.height < c[7]) {
            items_OK7.push_back(item);
        }
    }

    std::sort(items_OK7.begin(),items_OK7.end(),cmp);

    double sum7 = 0;

    int i7 = 0;

    std::cout << "Read " << items_OK7.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK7) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum7 += item.length * item.width * item.height * item.qty;
        i7 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i7 > 1000) {
            break;
        }
    }

    std::cout << sum7;


    std::vector<Item> items_OK8;

    for (auto& item : items) {
        if (item.length < a[8] && item.width < b[8] && item.height < c[8]) {
            items_OK8.push_back(item);
        }
    }

    std::sort(items_OK8.begin(),items_OK8.end(),cmp);

    double sum8 = 0;

    int i8 = 0;

    std::cout << "Read " << items_OK8.size() << " itemss_OK from the CSV file:" << std::endl;
    std::cout << "sta_code\tsku_code\tLength(CM)\tWidth(CM)\tHeight(CM)\tQty"
                 "\tvolume"<< std::endl;
    for (auto& item : items_OK8) {
        std::cout << item.sta_code << "\t"
                  << item.sku_code << "\t"
                  << item.length << "\t"
                  << item.width << "\t"
                  << item.height << "\t"
                  << item.qty << "\t"
                  << item.length * item.width * item.height * item.qty<< std::endl;
        sum8 += item.length * item.width * item.height * item.qty;
        i8 += item.qty;
        Item is;
        is.sta_code = item.sta_code;
        is.sku_code = item.sku_code;
        is.length = item.length;
        is.height = item.height;
        is.width = item.width;
        is.qty = item.qty;
        auto it = std::find(items.begin(), items.end(),is);
        items.erase(it);
        if (i8 > 1000) {
            break;
        }
    }

    std::cout << sum8;

    return 0;
}
