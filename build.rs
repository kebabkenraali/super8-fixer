fn main() {
    println!("cargo:rerun-if-changed=assets/super8fixer.ico");
    let mut r = winresource::WindowsResource::new();
    r.set_icon("assets/super8fixer.ico");
    r.set("FileDescription", "Super8 Fixer - frame-line slip repair and stabilization for digitized film");
    r.set("ProductName", "Super8 Fixer");
    r.compile().expect("failed to embed the icon");
}
