#import "@preview/ctheorems:1.1.3": *
#import "@preview/fontawesome:0.4.0": *

#let fontsizes = (
  normal: 12pt,
  section: 14pt,
  subsection: 12pt,
  large: 20pt,
  small: 8pt,
)

#let solarized = (
  yellow: rgb("#b58900"),
  orange: rgb("#cb4b17"),
  red: rgb("#dc322f"),
  magenta: rgb("#d33682"),
  violet: rgb("#6c71c4"),
  blue: rgb("#268bd2"),
  cyan: rgb("#2aa198"),
  cyanlight: rgb("#d4ecea"),
  green: rgb("#859900"),
  base2: rgb("#eee8d5"),
  gray: rgb("#f2f2f2"),
)

#let conf(doc) = {

    set text(
        font: "TeX Gyre Pagella",
        size: 12pt
    )
    show math.equation: set text(font: "TeX Gyre Pagella Math")
    show raw: set text(font: "TeX Gyre Pagella Math")

  // show link: set text(fill: gray)
  // show link: underline

  set align(left)
  show: thmrules.with(qed-symbol: $square$)

  // ! headers
  set heading(numbering: "1.1")

     set heading(numbering: "1.1")

      show heading.where(
        level: 1,
      ): it => text(
        size: fontsizes.section,
        weight: "bold",
        if (it.numbering != none) {
          par(
            leading: 0em,
            first-line-indent: 0em,
            counter(heading).display(it.numbering) + h(.5em) + smallcaps(it.body) + "\n",
          )
        } else {
          par(leading: 0em, first-line-indent: 0em, it.body + [.] + "\n")
        },
      )

      show heading.where(
        level: 2,
      ): it => text(
        size: fontsizes.subsection,
        weight: "semibold",
        // style: "italic",
        par(leading: 0em, first-line-indent: 0em, counter(heading).display(it.numbering) + h(.5em) + it.body),
        // it.numbering + h(.5em) + it.body + [.],
      )
      show heading.where(
        level: 3,
      ): it => text(
        size: fontsizes.subsection,
        weight: "semibold",
        // style: "italic",
        par(leading: 0em, first-line-indent: 0em, counter(heading).display(it.numbering) + h(.5em) + it.body),
        // it.numbering + h(.5em) + it.body + [.],
      )

      doc
}


// ! theorems
#let thmsettings = (
  inset: (top: 0.6em, left: .5em, right: .5em, bottom: 0.82em),
  base_level: 2,
  padding: (top: 0pt, bottom: -4pt),
)

#let theorem = thmbox(
  "theorem", // identifier
  text("Theorem"),//, fill: solarized.red), // head
  fill: solarized.gray,
  inset: thmsettings.inset,
  // stroke: 1pt
  base_level: thmsettings.base_level,
  supplement: "Theorem",
  padding: thmsettings.padding,
)

#let lemma = thmbox(
  "lemma", // identifier
  text("Lemma"), // head
  fill: solarized.gray,
  inset: thmsettings.inset,
  base_level: thmsettings.base_level,
  supplement: "Lemma",
  padding: thmsettings.padding,
)

#let proposition = thmbox(
  "proposition", // identifier
  // $arrow.hook$+" Proposition",
  text("Proposition"),//, fill: solarized.magenta), // head
  fill: solarized.gray,
  inset: thmsettings.inset,
  base_level: thmsettings.base_level,
  supplement: "Prop.",
  padding: thmsettings.padding,
  // stroke: 1pt
)

#let corollary = thmbox(
  "corollary",
  // $arrow.hook$+" Corollary",
  text("Corollary", fill: solarized.orange), // head
  fill: solarized.gray,
  inset: thmsettings.inset,
  base_level: thmsettings.base_level,
  padding: thmsettings.padding,
  supplement: "Corollary",
)

#let definition = thmbox(
  "definition",
  "Definition",
  text("Definition", fill: solarized.gray),
  fill: solarized.gray,
  inset: thmsettings.inset,
  base_level: thmsettings.base_level,
  padding: thmsettings.padding,
)

#let example = thmbox(
  "example",
  "Example",
  fill: solarized.cyanlight,
  inset: thmsettings.inset,
  padding: thmsettings.padding,
  base_level: thmsettings.base_level,
)


#let exercise = thmbox(
  "exercise",
  "Exercise",
  fill: solarized.cyanlight,
  inset: thmsettings.inset,
  padding: thmsettings.padding,
  base_level: thmsettings.base_level,
)

#let remark = thmbox(
  "remark",
  "Remark",
  stroke: none,
  inset: (top: 0.4em, left: .5em, right: .5em, bottom: 0.6em),
  base_level: 2,
  padding: thmsettings.padding,
)


#let axiom = thmbox(
  "axiom",
  "Axiom",
  base_level: 0,
  fill: solarized.gray,
  inset: thmsettings.inset,
)

#let proof = thmproof(
  "proof",
  text(
    smallcaps("Proof"),
    // highlight("Proof", fill: white, stroke: black, top-edge: "cap-height", extent: 3pt),
    style: "oblique",
    weight: "regular",
  ),

  inset: (top: 0em, left: 2.8em, right: 1.4em),
  separator: [#h(0.1em). #h(0.2em)],
)

#let solution = thmproof(
  "solution",
  text(
    smallcaps("Solution"),
    // highlight("Proof", fill: white, stroke: black, top-edge: "cap-height", extent: 3pt),
    style: "oblique",
    weight: "regular",
  ),

  inset: (top: 0em, left: 2.8em, right: 1.4em),
  separator: [#h(0.1em). #h(0.2em)],
)
