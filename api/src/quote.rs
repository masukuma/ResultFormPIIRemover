//! What a purchase of form packages costs (PRICING.md): the packages, a
//! card-fee line sized so that the operator keeps the packages' whole price
//! after the card processor's fee, and VAT on both lines. Integer cents
//! throughout; rates are in basis points (1/100 of a percent, so 1600 is
//! 16%). Every share of an amount is rounded half up to the cent.

use serde_json::{Value, json};

/// Basis points in a whole.
pub const BASIS: i64 = 10_000;

/// The rates a quote is made with (configuration, api/README.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rates {
    /// Forms in a package.
    pub package_forms: i64,
    /// A package's price, before the card fee and VAT.
    pub package_price_cents: i64,
    /// VAT, charged on every line.
    pub vat_basis_points: i64,
    /// The card processor's fee: this share of the whole charge…
    pub card_fee_basis_points: i64,
    /// …plus this much per charge.
    pub card_fee_fixed_cents: i64,
}

/// `amount × basis_points / 10,000`, rounded half up to the cent; for
/// amounts of zero or more.
pub fn share(amount: i64, basis_points: i64) -> i64 {
    ((i128::from(amount) * i128::from(basis_points) + i128::from(BASIS / 2)) / i128::from(BASIS)) as i64
}

/// The smallest whole number at or above `num / den` (`den > 0`).
fn ceil_div(num: i128, den: i128) -> i128 {
    num.div_euclid(den) + i128::from(num.rem_euclid(den) != 0)
}

impl Rates {
    /// Whether the rates make a quote possible: a package of 1 to 1,000,000
    /// forms and 1 cent to $1,000,000, VAT from 0 to 100%, and a processor's
    /// rate below 50% (so that each cent of card fee covers its own share
    /// of the fee) with a fixed fee of at most $1,000,000.
    pub fn check(&self) -> Result<(), String> {
        const MILLION: i64 = 1_000_000;
        if !(1..=MILLION).contains(&self.package_forms) || !(1..=100 * MILLION).contains(&self.package_price_cents) {
            return Err("PACKAGE_FORMS must be from 1 to 1000000, and PACKAGE_PRICE_CENTS from 1 to 100000000".into());
        }
        if !(0..=BASIS).contains(&self.vat_basis_points) {
            return Err("VAT_RATE_BASIS_POINTS must be from 0 to 10000".into());
        }
        if !(0..BASIS / 2).contains(&self.card_fee_basis_points) || !(0..=100 * MILLION).contains(&self.card_fee_fixed_cents) {
            return Err("CARD_FEE_BASIS_POINTS must be from 0 to 4999, and CARD_FEE_FIXED_CENTS from 0 to 100000000".into());
        }
        Ok(())
    }

    /// VAT on a line of `amount` cents.
    pub fn vat(&self, amount: i64) -> i64 {
        share(amount, self.vat_basis_points)
    }

    /// The processor's fee on a charge of `total` cents: its rate of the
    /// total, rounded, and its fixed part.
    pub fn processor_fee(&self, total: i64) -> i64 {
        share(total, self.card_fee_basis_points) + self.card_fee_fixed_cents
    }

    /// The total of a purchase of `net` cents of packages and a card fee of
    /// `card_fee` cents: both lines and the VAT on each.
    pub fn total(&self, net: i64, card_fee: i64) -> i64 {
        net + card_fee + self.vat(net) + self.vat(card_fee)
    }

    /// What the operator keeps of that purchase: the total less the VAT on
    /// each line (owed to the tax authority) and the processor's fee.
    pub fn kept(&self, net: i64, card_fee: i64) -> i64 {
        let total = self.total(net, card_fee);
        total - self.vat(net) - self.vat(card_fee) - self.processor_fee(total)
    }

    /// The card fee in closed form, ignoring rounding:
    /// ceil(((1 + v)·p·B + f) / (1 − (1 + v)·p)), with v the VAT rate, p and
    /// f the processor's rate and fixed fee, and B the packages' price.
    pub fn card_fee_estimate(&self, net: i64) -> i64 {
        let (v, p, f, b, basis) =
            (i128::from(self.vat_basis_points), i128::from(self.card_fee_basis_points), i128::from(self.card_fee_fixed_cents), i128::from(net), i128::from(BASIS));
        let num = (basis + v) * p * b + f * basis * basis;
        let den = basis * basis - (basis + v) * p;
        ceil_div(num, den) as i64
    }

    /// The card fee for `net` cents of packages: the smallest whole number
    /// of cents with which the operator keeps at least `net`. The closed
    /// form, moved a cent at a time until it is that smallest one (rounding
    /// can put it a cent or so off). What the operator keeps less `net` is
    /// the card fee less the processor's fee, which never falls as the card
    /// fee grows (each cent adds at most one cent of fee while the
    /// processor's rate is below 50%), so the search ends at the minimum.
    pub fn card_fee(&self, net: i64) -> i64 {
        let covers = |fee: i64| self.kept(net, fee) >= net;
        let mut fee = self.card_fee_estimate(net).max(0);
        while fee > 0 && covers(fee - 1) {
            fee -= 1;
        }
        while !covers(fee) {
            fee += 1;
        }
        fee
    }

    /// The quote for `packages` packages bought by card.
    pub fn quote(&self, packages: i64) -> Quote {
        let net = packages * self.package_price_cents;
        self.quote_with_fee(packages, self.card_fee(net))
    }

    /// The quote for `packages` packages paid outside the card processor
    /// (a bank transfer): no card fee, VAT on the packages.
    pub fn quote_without_card_fee(&self, packages: i64) -> Quote {
        self.quote_with_fee(packages, 0)
    }

    fn quote_with_fee(&self, packages: i64, card_fee: i64) -> Quote {
        let net = packages * self.package_price_cents;
        let total = self.total(net, card_fee);
        let processor_fee = if card_fee > 0 { self.processor_fee(total) } else { 0 };
        Quote {
            packages,
            forms: packages * self.package_forms,
            package_forms: self.package_forms,
            package_price_cents: self.package_price_cents,
            net_cents: net,
            card_fee_cents: card_fee,
            vat_basis_points: self.vat_basis_points,
            vat_net_cents: self.vat(net),
            vat_card_fee_cents: self.vat(card_fee),
            vat_cents: self.vat(net) + self.vat(card_fee),
            total_cents: total,
            processor_fee_cents: processor_fee,
            kept_cents: total - self.vat(net) - self.vat(card_fee) - processor_fee,
        }
    }

    /// The price of one form in a package, in micro-dollars, rounded.
    pub fn price_per_form_microusd(&self) -> i64 {
        per_form_microusd(self.package_price_cents, self.package_forms)
    }

    /// The card fee as people write it: "4.4% + $0.30".
    pub fn card_fee_terms(&self) -> String {
        format!("{} + ${}", percent(self.card_fee_basis_points), format_cents(self.card_fee_fixed_cents))
    }
}

/// `cents` shared over `forms`, in micro-dollars, rounded half up.
fn per_form_microusd(cents: i64, forms: i64) -> i64 {
    ((i128::from(cents) * 10_000 * 2 + i128::from(forms)) / (2 * i128::from(forms.max(1)))) as i64
}

/// What a purchase costs, line by line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quote {
    pub packages: i64,
    pub forms: i64,
    pub package_forms: i64,
    pub package_price_cents: i64,
    /// The packages: `packages × package_price_cents`.
    pub net_cents: i64,
    /// The card fee line (0 when paid outside the card processor).
    pub card_fee_cents: i64,
    pub vat_basis_points: i64,
    pub vat_net_cents: i64,
    pub vat_card_fee_cents: i64,
    pub vat_cents: i64,
    /// What the customer pays.
    pub total_cents: i64,
    /// The processor's fee on the total, as configured (operator's figure).
    pub processor_fee_cents: i64,
    /// What the operator keeps: the total less VAT and the processor's fee.
    pub kept_cents: i64,
}

impl Quote {
    /// The lines, as a tax invoice lists them: the packages, then the card
    /// fee if there is one, each with its VAT.
    pub fn lines(&self, card_fee_terms: Option<&str>) -> Vec<Value> {
        let mut lines = vec![json!({
            "item": "packages",
            "description": format!("{} package(s) of {} forms", self.packages, self.package_forms),
            "quantity": self.packages,
            "unit_amount_cents": self.package_price_cents,
            "amount_cents": self.net_cents,
            "vat_rate_basis_points": self.vat_basis_points,
            "vat_cents": self.vat_net_cents,
        })];
        if self.card_fee_cents > 0 {
            let description = match card_fee_terms {
                Some(t) => format!("card processing fee (the processor's {t} of the charge)"),
                None => "card processing fee".into(),
            };
            lines.push(json!({
                "item": "card_fee",
                "description": description,
                "quantity": 1,
                "unit_amount_cents": self.card_fee_cents,
                "amount_cents": self.card_fee_cents,
                "vat_rate_basis_points": self.vat_basis_points,
                "vat_cents": self.vat_card_fee_cents,
            }));
        }
        lines
    }

    /// The quote as the API returns it (`GET /v1/billing/quote`).
    pub fn to_value(&self, rates: &Rates) -> Value {
        json!({
            "object": "quote",
            "currency": "usd",
            "packages": self.packages,
            "forms": self.forms,
            "lines": self.lines(Some(&rates.card_fee_terms())),
            "net_cents": self.net_cents,
            "card_fee_cents": self.card_fee_cents,
            "vat_rate_basis_points": self.vat_basis_points,
            "vat_net_cents": self.vat_net_cents,
            "vat_card_fee_cents": self.vat_card_fee_cents,
            "vat_cents": self.vat_cents,
            "total_cents": self.total_cents,
            "total_usd": format_cents(self.total_cents),
            "effective_price_per_form_usd": crate::billing::format_usd(per_form_microusd(self.total_cents, self.forms)),
        })
    }
}

/// Cents as dollars, with two decimals (4315 → "43.15").
pub fn format_cents(cents: i64) -> String {
    let sign = if cents < 0 { "-" } else { "" };
    let c = cents.unsigned_abs();
    format!("{sign}{}.{:02}", c / 100, c % 100)
}

/// Basis points as a percentage (440 → "4.4%", 1600 → "16%").
pub fn percent(basis_points: i64) -> String {
    let whole = basis_points / 100;
    let frac = format!("{:02}", basis_points.rem_euclid(100));
    let frac = frac.trim_end_matches('0');
    if frac.is_empty() { format!("{whole}%") } else { format!("{whole}.{frac}%") }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults: $35 a package of 100, VAT 16%, a card fee of 4.4% + $0.30.
    fn rates() -> Rates {
        Rates { package_forms: 100, package_price_cents: 3500, vat_basis_points: 1600, card_fee_basis_points: 440, card_fee_fixed_cents: 30 }
    }

    /// The smallest card fee found by counting up from nothing.
    fn smallest_by_search(r: &Rates, net: i64) -> i64 {
        (0..).find(|&s| r.kept(net, s) >= net).unwrap()
    }

    #[test]
    fn one_package_costs_43_15_and_the_operator_keeps_35_00() {
        let q = rates().quote(1);
        assert_eq!((q.net_cents, q.card_fee_cents), (3500, 220));
        assert_eq!((q.vat_net_cents, q.vat_card_fee_cents, q.vat_cents), (560, 35, 595));
        assert_eq!(q.total_cents, 4315);
        // the processor takes round(4.4% of 43.15 + 0.30) = 2.20: exactly the card fee
        assert_eq!((q.processor_fee_cents, q.kept_cents), (220, 3500));
        assert_eq!(q.forms, 100);
    }

    #[test]
    fn two_ten_and_a_hundred_packages() {
        let q = rates().quote(2);
        assert_eq!((q.net_cents, q.card_fee_cents, q.vat_net_cents, q.vat_card_fee_cents, q.total_cents), (7000, 408, 1120, 65, 8593));
        let q = rates().quote(10);
        assert_eq!((q.net_cents, q.card_fee_cents, q.vat_net_cents, q.vat_card_fee_cents, q.total_cents), (35000, 1914, 5600, 306, 42820));
        assert_eq!(q.vat_cents, 5906);
        assert_eq!(q.kept_cents, 35000);
        let q = rates().quote(100);
        assert_eq!((q.net_cents, q.card_fee_cents, q.vat_net_cents, q.vat_card_fee_cents, q.total_cents), (350000, 18856, 56000, 3017, 427873));
        assert_eq!((q.forms, q.kept_cents), (10_000, 350000));
    }

    #[test]
    fn the_closed_form_is_moved_to_the_smallest_fee_that_covers() {
        let r = rates();
        // rounding puts the closed form a cent above the minimum for 10 packages
        assert_eq!(r.card_fee_estimate(35000), 1915);
        assert_eq!(r.card_fee(35000), 1914);
        assert!(r.kept(35000, 1914) >= 35000 && r.kept(35000, 1913) < 35000);
        assert_eq!(r.card_fee_estimate(3500), 220);
    }

    #[test]
    fn the_card_fee_is_the_smallest_that_covers_the_processor_for_any_rates() {
        let others = [
            rates(),
            Rates { card_fee_basis_points: 290, ..rates() },
            Rates { card_fee_basis_points: 0, card_fee_fixed_cents: 0, ..rates() },
            Rates { vat_basis_points: 0, ..rates() },
            Rates { vat_basis_points: 1650, card_fee_basis_points: 4999, card_fee_fixed_cents: 1, ..rates() },
            Rates { package_price_cents: 1999, package_forms: 7, vat_basis_points: 2500, card_fee_basis_points: 350, card_fee_fixed_cents: 25 },
        ];
        for r in others {
            for n in 1..=100 {
                let net = n * r.package_price_cents;
                let fee = r.card_fee(net);
                assert_eq!(fee, smallest_by_search(&r, net), "{r:?}, {n} packages");
                assert!(r.kept(net, fee) >= net, "{r:?}, {n}");
                assert!(fee == 0 || r.kept(net, fee - 1) < net, "{r:?}, {n}: {fee} is not the smallest");
                let q = r.quote(n);
                assert_eq!(q.total_cents, q.net_cents + q.card_fee_cents + q.vat_cents);
                // at the smallest fee the operator keeps the packages' price, to the cent
                assert_eq!(q.kept_cents, q.net_cents, "{r:?}, {n}");
            }
        }
        // no processor fee: no card fee
        assert_eq!(Rates { card_fee_basis_points: 0, card_fee_fixed_cents: 0, ..rates() }.quote(3).card_fee_cents, 0);
    }

    #[test]
    fn shares_are_rounded_half_up_to_the_cent() {
        assert_eq!(share(1, 5000), 1); // 0.5
        assert_eq!(share(3, 5000), 2); // 1.5
        assert_eq!(share(2, 2500), 1); // 0.5
        assert_eq!(share(1, 4999), 0);
        assert_eq!(share(219, 1600), 35); // 35.04
        assert_eq!(share(1914, 1600), 306); // 306.24
        assert_eq!(share(18856, 1600), 3017); // 3016.96
        // 4.4% of 125 is 5.5: the fee is 6 + 30
        assert_eq!(rates().processor_fee(125), 36);
        assert_eq!(rates().processor_fee(4315), 220); // 189.86 + 30
        assert_eq!(share(i64::MAX / 10_000, 10_000), i64::MAX / 10_000);
    }

    #[test]
    fn a_bank_transfer_has_vat_and_no_card_fee() {
        let q = rates().quote_without_card_fee(2);
        assert_eq!((q.net_cents, q.card_fee_cents, q.vat_cents, q.total_cents, q.kept_cents), (7000, 0, 1120, 8120, 7000));
        assert_eq!(q.lines(None).len(), 1);
    }

    #[test]
    fn the_quote_lists_each_line_with_its_vat() {
        let r = rates();
        let v = r.quote(1).to_value(&r);
        assert_eq!(
            v["lines"][0],
            json!({"item": "packages", "description": "1 package(s) of 100 forms", "quantity": 1, "unit_amount_cents": 3500, "amount_cents": 3500, "vat_rate_basis_points": 1600, "vat_cents": 560})
        );
        assert_eq!((v["lines"][1]["item"].as_str(), v["lines"][1]["amount_cents"].as_i64(), v["lines"][1]["vat_cents"].as_i64()), (Some("card_fee"), Some(220), Some(35)));
        assert!(v["lines"][1]["description"].as_str().unwrap().contains("4.4% + $0.30"));
        assert_eq!((v["total_usd"].as_str(), v["effective_price_per_form_usd"].as_str()), (Some("43.15"), Some("0.4315")));
        assert_eq!(r.quote(10).to_value(&r)["effective_price_per_form_usd"], "0.4282");
        assert_eq!(r.price_per_form_microusd(), 350_000);
    }

    #[test]
    fn rates_are_checked() {
        assert!(rates().check().is_ok());
        assert!(Rates { card_fee_basis_points: 5000, ..rates() }.check().is_err());
        assert!(Rates { vat_basis_points: 10_001, ..rates() }.check().is_err());
        assert!(Rates { package_forms: 0, ..rates() }.check().is_err());
        assert!(Rates { card_fee_fixed_cents: -1, ..rates() }.check().is_err());
    }

    #[test]
    fn money_and_rates_are_written_as_people_write_them() {
        assert_eq!((format_cents(4315), format_cents(30), format_cents(0), format_cents(-5)), ("43.15".into(), "0.30".into(), "0.00".into(), "-0.05".into()));
        assert_eq!((percent(440), percent(1600), percent(25), percent(1650)), ("4.4%".into(), "16%".into(), "0.25%".into(), "16.5%".into()));
        assert_eq!(rates().card_fee_terms(), "4.4% + $0.30");
    }
}
