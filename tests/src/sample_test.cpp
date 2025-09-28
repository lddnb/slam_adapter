#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sample Test", "[sample]") {
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("Another Sample Test", "[sample]") {
    int x = 2;
    int y = 3;
    REQUIRE(x * y == 6);
    REQUIRE(y - x == 1);
}
