for pkg in ["Utils"]
    ispath(Pkg.dir(pkg)) && (Pkg.checkout(pkg); continue)
    Pkg.clone("https://github.com/AStupidBear/$pkg.jl.git")
    Pkg.build(pkg)
end
