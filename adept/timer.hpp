struct timer{
double total_time = 0;
std::chrono::time_point<std::chrono::high_resolution_clock> time_begin;
std::chrono::time_point<std::chrono::high_resolution_clock> time_end;
std::chrono::duration<double> diff;

void timer_start(){
    time_begin = std::chrono::high_resolution_clock::now();
}

void timer_end(){
    time_end = std::chrono::high_resolution_clock::now();
}

double get_total_time(){
    diff = time_end - time_begin;
    return diff.count();
}


};