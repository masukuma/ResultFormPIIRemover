//! What the templates don't say about each printing, for the synthetic forms
//! and the teacher's labels: where names, parties and dates are written
//! (templates deliberately have no zones for them: they stay visible), the
//! declaration areas, where stamps usually go, and the printed columns a
//! template's zone doesn't sit on. Coordinates are pixels in the template's
//! reference image, read off the references.

/// What is written in a cell that is not a template zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// a person's name (class `name`)
    Name,
    /// a party's name or "INDEPENDENT": no one's personal data
    Party,
    /// a date: digits that are not an ID or a phone number
    Date,
    /// a count written in a results cell (protected): digits again
    Count,
}

/// Cells of one kind, `rows` of them `step` px apart, holding `role`'s
/// writing (agent, officer); `zone` names them in the labels.
#[derive(Clone, Copy, Debug)]
pub struct Cells {
    pub kind: Kind,
    pub zone: &'static str,
    pub rect: [f64; 4],
    pub rows: usize,
    pub step: f64,
    pub role: &'static str,
}

/// A printing's extra layout.
pub struct Layout {
    pub template: &'static str,
    pub cells: &'static [Cells],
    /// the officers' declaration areas (class `declaration`)
    pub declaration: &'static [[f64; 4]],
    /// where a stamp's centre usually is
    pub stamp_spots: &'static [[f64; 2]],
    /// table zones whose rect isn't on the printed column it is named for:
    /// (zone, the printed column's x0, x1). The synthetic forms write there,
    /// as people do.
    pub columns: &'static [(&'static str, f64, f64)],
    /// what the stamp's top ring says
    pub stamp_title: &'static str,
}

const fn c(kind: Kind, zone: &'static str, rect: [f64; 4], rows: usize, step: f64, role: &'static str) -> Cells {
    Cells { kind, zone, rect, rows, step, role }
}

pub const LAYOUTS: &[Layout] = &[
    Layout {
        template: "form34a_2022",
        cells: &[
            c(Kind::Name, "agent_name", [135.0, 1569.0, 360.0, 36.3], 11, 36.3, "agent"),
            c(Kind::Party, "agent_party", [682.0, 1572.0, 362.0, 36.4], 11, 36.4, "agent"),
            c(Kind::Date, "agent_date", [1450.0, 1584.0, 110.0, 36.5], 11, 36.5, "agent"),
            c(Kind::Name, "presiding_officer_name", [300.0, 1352.0, 590.0, 50.0], 1, 56.0, "officer"),
            c(Kind::Name, "deputy_officer_name", [380.0, 1408.0, 510.0, 50.0], 1, 56.0, "officer"),
            c(Kind::Date, "officer_date", [1345.0, 1352.0, 210.0, 50.0], 2, 56.0, "officer"),
        ],
        declaration: &[[60.0, 1250.0, 1540.0, 232.0]],
        stamp_spots: &[[1180.0, 1395.0], [1050.0, 1420.0], [1300.0, 2060.0], [900.0, 1650.0]],
        columns: &[],
        stamp_title: "PRESIDING OFFICER",
    },
    Layout {
        template: "form34b_2022_block",
        cells: &[
            c(Kind::Name, "agent_name", [108.0, 160.0, 424.0, 81.0], 4, 83.3, "agent"),
            c(Kind::Party, "agent_party", [818.0, 160.0, 310.0, 81.0], 4, 83.3, "agent"),
            c(Kind::Date, "agent_date", [1846.0, 160.0, 228.0, 81.0], 4, 83.3, "agent"),
            c(Kind::Name, "ro_name", [596.0, 526.0, 450.0, 78.0], 1, 0.0, "officer"),
            c(Kind::Date, "ro_date", [130.0, 775.0, 950.0, 80.0], 1, 0.0, "officer"),
            c(Kind::Count, "form34a_submitted", [416.0, 1135.0, 600.0, 75.0], 1, 0.0, "officer"),
            c(Kind::Count, "form34a_received", [1476.0, 1135.0, 560.0, 75.0], 1, 0.0, "officer"),
            c(Kind::Name, "handover_ro_name", [416.0, 1272.0, 600.0, 85.0], 1, 0.0, "officer"),
            c(Kind::Name, "chair_name", [1476.0, 1272.0, 560.0, 85.0], 1, 0.0, "officer"),
            c(Kind::Date, "handover_date", [416.0, 1700.0, 600.0, 80.0], 1, 0.0, "officer"),
            c(Kind::Date, "chair_date", [1476.0, 1700.0, 560.0, 80.0], 1, 0.0, "officer"),
        ],
        declaration: &[[30.0, 520.0, 1080.0, 345.0], [30.0, 1005.0, 2080.0, 965.0]],
        stamp_spots: &[[1000.0, 700.0], [900.0, 1600.0], [1950.0, 1600.0], [1300.0, 300.0]],
        columns: &[],
        stamp_title: "RETURNING OFFICER",
    },
    Layout {
        template: "form34b_2022_block_b",
        cells: &[
            c(Kind::Name, "agent_name", [388.0, 880.0, 752.0, 51.0], 5, 51.6, "agent"),
            c(Kind::Party, "agent_party", [1368.0, 880.0, 304.0, 51.0], 5, 51.6, "agent"),
            c(Kind::Date, "agent_date", [2210.0, 880.0, 310.0, 51.0], 5, 51.6, "agent"),
            c(Kind::Name, "ro_name", [430.0, 1160.0, 660.0, 58.0], 1, 0.0, "officer"),
            c(Kind::Date, "ro_date", [140.0, 1352.0, 470.0, 62.0], 1, 0.0, "officer"),
            c(Kind::Count, "valid_votes", [1145.0, 113.0, 210.0, 48.0], 4, 50.5, "officer"),
        ],
        declaration: &[[25.0, 1160.0, 1250.0, 270.0]],
        stamp_spots: &[[1300.0, 1300.0], [900.0, 1330.0], [2000.0, 1000.0]],
        columns: &[],
        stamp_title: "RETURNING OFFICER",
    },
    Layout {
        template: "form34b_2022_block_c",
        cells: &[
            c(Kind::Name, "agent_name", [368.0, 202.0, 678.0, 69.0], 3, 69.0, "agent"),
            c(Kind::Party, "agent_party", [1280.0, 202.0, 238.0, 69.0], 3, 69.0, "agent"),
            c(Kind::Date, "agent_date", [2167.0, 202.0, 315.0, 69.0], 3, 69.0, "agent"),
            c(Kind::Name, "ro_name", [830.0, 445.0, 680.0, 57.0], 1, 0.0, "officer"),
            c(Kind::Date, "ro_date", [140.0, 610.0, 1360.0, 52.0], 1, 0.0, "officer"),
            c(Kind::Name, "chair_name", [1680.0, 955.0, 640.0, 55.0], 1, 0.0, "officer"),
        ],
        declaration: &[[20.0, 465.0, 1500.0, 200.0], [15.0, 805.0, 2470.0, 670.0]],
        stamp_spots: &[[1700.0, 560.0], [900.0, 1200.0], [2100.0, 1200.0]],
        // The printed columns (rules at x 1520, 1839, 2159, 2484): "Tel.
        // Contact" is 1524-1839 and "Signature" 1843-2159; the template's
        // agent_phone rect (1844-2158) is on the Signature column and its
        // agent_signatures rect (2164-2483) on the Date column.
        columns: &[("agent_phone", 1526.0, 1837.0), ("agent_signatures", 1846.0, 2157.0)],
        stamp_title: "RETURNING OFFICER",
    },
];

pub fn layout(template: &str) -> Option<&'static Layout> {
    LAYOUTS.iter().find(|l| l.template == template)
}

impl Cells {
    /// Each row's rect.
    pub fn rows(&self) -> Vec<[f64; 4]> {
        (0..self.rows).map(|r| [self.rect[0], self.rect[1] + r as f64 * self.step, self.rect[2], self.rect[3]]).collect()
    }

    /// The name of row `r` in the labels (as the template names zone rows).
    pub fn name(&self, r: usize) -> String {
        if self.rows > 1 { format!("{}[{}]", self.zone, r) } else { self.zone.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::prelude::*;
    use std::path::Path;

    #[test]
    fn every_layout_is_inside_its_reference_and_names_a_template() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../templates");
        for l in LAYOUTS {
            let t = crate::templates::load_template(&root.join(l.template)).unwrap();
            let (w, h) = (t.image.cols() as f64, t.image.rows() as f64);
            let inside = |r: [f64; 4]| r[0] >= 0.0 && r[1] >= 0.0 && r[0] + r[2] <= w && r[1] + r[3] <= h;
            for c in l.cells {
                assert!(c.rows().into_iter().all(inside), "{} {}", l.template, c.zone);
            }
            assert!(l.declaration.iter().all(|&r| inside(r)));
            for (zone, x0, x1) in l.columns {
                assert!(t.zones.iter().any(|z| z.key() == *zone), "{zone}");
                assert!(x0 < x1 && *x1 <= w);
            }
        }
        assert!(layout("form34b_2022_results").is_none());
    }

}
