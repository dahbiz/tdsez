#include "tdsez_parser.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::string& expected, const std::function<void()>& check)
{
    try {
        check();
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "unexpected validation error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected validation error containing: " + expected);
}

} // namespace

int main()
{
    try {
        // Keep this test independent of eigensolver, filesystem, and MPI setup.
        TDSEZParser::EnablePropagation = PETSC_FALSE;
        TDSEZParser::Potential = "0.5*x*x";
        TDSEZParser::Mass = "1.0";
        TDSEZParser::NBoundStates = 3;
        TDSEZParser::initParsers();

        TDSEZParser::ValidateOrThrow();
        require(TDSEZParser::InitialStateMode == 0, "ground state mode");

        TDSEZParser::InitialState = "state:2";
        TDSEZParser::ValidateOrThrow();
        require(TDSEZParser::InitialStateMode == 1 &&
                TDSEZParser::InitialStateIndex == 2, "single state selection");

        TDSEZParser::InitialState = "sup: 0.7*0 + -0.4*2";
        TDSEZParser::ValidateOrThrow();
        require(TDSEZParser::InitialStateMode == 2 &&
                TDSEZParser::InitialStateTerms.size() == 2 &&
                TDSEZParser::InitialStateTerms[1].first == 2 &&
                std::abs(TDSEZParser::InitialStateTerms[1].second + 0.4) < 1e-12,
                "superposition terms");

        TDSEZParser::InitialState = "state:3";
        rejects("only NBoundStates=3", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::InitialState = "state:1typo";
        rejects("non-integer index", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::InitialState = "sup: 0.5extra*0 + 0.5*1";
        rejects("non-numeric coefficient", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::InitialState = "ground";

        TDSEZParser::Dimension = 4;
        rejects("Dimension must be 1, 2 or 3", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::Dimension = 1;

        TDSEZParser::LMaxX = std::numeric_limits<PetscReal>::quiet_NaN();
        rejects("X domain bound is not finite", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::LMaxX = 10.0;

        TDSEZParser::LMaxX = TDSEZParser::LMinX;
        rejects("X requires LMax > LMin", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::LMaxX = 10.0;

        TDSEZParser::MassExpr.SetExpr("0.0");
        rejects("Mass must be > 0", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::MassExpr.SetExpr("1.0");

        TDSEZParser::Dimension = 2;
        TDSEZParser::MassExpr.SetExpr("1.0 - 0.02*y*y");
        rejects("Mass must be > 0", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::MassExpr.SetExpr("1.0");

        TDSEZParser::VPot.SetExpr("sqrt(1.0-y*y)");
        rejects("Potential", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::VPot.SetExpr("0.5*x*x");

        TDSEZParser::MassExpr.SetExpr("missing_mass_symbol");
        rejects("Mass expression evaluation failed", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::MassExpr.SetExpr("1.0");

        TDSEZParser::Dimension = 3;
        TDSEZParser::MassExpr.SetExpr("1.0 - 0.02*z*z");
        rejects("Mass must be > 0", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::MassExpr.SetExpr("1.0");
        TDSEZParser::Dimension = 1;

        TDSEZParser::EnablePropagation = PETSC_TRUE;
        TDSEZParser::TimeStep = 0.0;
        rejects("TimeStep and FinalTime must be > 0", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::TimeStep = std::numeric_limits<PetscReal>::quiet_NaN();
        rejects("non-finite TimeStep/FinalTime", [] { TDSEZParser::ValidateOrThrow(); });
        TDSEZParser::TimeStep = 0.01;
        TDSEZParser::EnablePropagation = PETSC_FALSE;

        TDSEZParser::BoundaryType = "wall";
        TDSEZParser::Polarization = "XY";
        TDSEZParser::ValidateOrThrow();
        require(TDSEZParser::BoundaryType == "dirichlet" &&
                TDSEZParser::Polarization == "xy", "enum normalization");

        std::cout << "parser validation unit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "parser validation unit test failed: " << error.what() << '\n';
        return 1;
    }
}
