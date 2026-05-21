pub struct Foo {
    x: i64,
}

impl Foo {
    pub fn new(x: i64) -> Self {
        Self { x }
    }

    pub fn compute(&self) -> i64 {
        self.x + 1
    }
}
