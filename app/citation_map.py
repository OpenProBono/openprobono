import re

import spacy

from app.models import Citation


def split_overlapping_intervals(intervals: list) -> list:
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    # Find all unique points
    points = set()
    for start, end in intervals:
        points.add(start)
        points.add(end)
    points = sorted(points)

    # Create result intervals by checking which original intervals cover each segment
    result = []
    min_interval_len = 5
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        # Exclude small intervals
        if end - start < min_interval_len:
            continue
        # Check if this segment is covered by any interval
        for interval in intervals:
            if interval[0] <= start and interval[1] >= end:
                result.append((start, end))
                break

    return result

def extract_cited_clauses(text: str) -> list:
    nlp = spacy.load("en_core_web_lg")

    # Define a regex pattern to split on conjunctions and specific punctuation, capturing the split points
    # split_pattern = r"[;,\s]*(?:\b(?:and|or|but|so|yet|nor|for|although|because|if|when|while|whereas|since)\b)[\s,;]*|[;,]"

    # Split text into paragraphs
    paragraphs = text.split("\n\n")
    results = []
    citation_counts = {}

    for paragraph in paragraphs:
        # Find all citation groups in the paragraph
        citation_groups = re.findall(r"\[(\d+)\]", paragraph)

        if citation_groups:
            # Process each citation number and track instances
            citations = [int(n.strip()) for n in citation_groups]
            citation_objs = []
            for num in citations:
                citation_counts[num] = citation_counts.get(num, 0) + 1
                citation_objs.append(Citation(number=num, instance=citation_counts[num]))

            # Remove citations for easier processing
            # text_without_citations = re.sub(r"\s*\[\d+\]", "", paragraph)
            doc = nlp(paragraph)

            # Treat each sentence as a single unit to start with
            for sent in doc.sents:
                # Split sentence into sub-clauses based on conjunctions and punctuation
                # clauses = re.split(split_pattern, sent.text)

                # for clause_text in clauses:
                #     clause_text = clause_text.strip()
                #     if clause_text:
                results.append({
                    "clause": sent.text,
                    "citations": citation_objs,
                })
    return results

# Example usage:
# text = """In North Carolina, the rules and statutes related to counterclaims in civil procedures are outlined in Rule 13 of the North Carolina Rules of Civil Procedure. Here's a summary of the key points:

# 1. **Compulsory Counterclaims**: A party must state as a counterclaim any claim they have against an opposing party if it arises out of the same transaction or occurrence that is the subject matter of the opposing party's claim. However, this is not required if the claim was already the subject of another pending action or if the court did not acquire jurisdiction to render a personal judgment on that claim [1][2][3].

# 2. **Permissive Counterclaims**: A party may state a counterclaim against an opposing party even if it does not arise out of the transaction or occurrence that is the subject matter of the opposing party's claim [1][2][3].

# 3. **Counterclaim Exceeding Opposing Claim**: A counterclaim can seek relief that exceeds in amount or differs in kind from what the opposing party seeks [1][2][3].

# 4. **Counterclaims Against the State**: The rules do not expand the right to assert counterclaims or claim credit against the State of North Carolina beyond what is fixed by law [1][2][3].

# 5. **Counterclaims Maturing or Acquired After Pleading**: With the court's permission, a claim that matures or is acquired after the initial pleading can be presented as a counterclaim through a supplemental pleading [1][2][3].

# 6. **Omitted Counterclaims**: If a counterclaim is omitted due to oversight, inadvertence, or excusable neglect, or if justice requires, it may be set up by amendment with the court's leave [1][2][3].

# 7. **Crossclaims**: A party may state a crossclaim against a coparty if it arises out of the transaction or occurrence that is the subject matter of the original action or a counterclaim therein [1][2][3].

# 8. **Additional Parties**: The court may order additional parties to be brought in as defendants if their presence is required for complete relief in the determination of a counterclaim or crossclaim [1][2][3].

# 9. **Separate Trials and Judgments**: The court can order separate trials for counterclaims or crossclaims and render judgments accordingly, even if the opposing party's claims have been dismissed or otherwise disposed of [1][2][3].

# These rules are designed to ensure that all related claims are addressed in a single proceeding, promoting judicial efficiency and fairness."""

# context = """a. The responsive pleading shall be served within 20 days after notice of the court's action in ruling on the motion or postponing its disposition until the trial on the merits; b. If the court grants a motion for a more definite statement, the responsive pleading shall be served within 20 days after service of the more definite statement. (2) Cases Removed to United States District Court. - Upon the filing in a district court of the United States of a petition for the removal of a civil action or proceeding from a court in this State and the filing of a copy of the petition in the State court, the State court shall proceed no further therein unless and until the case is remanded. If it shall be finally determined in the United States courts that the action or proceeding was not removable or was improperly removed, or for other reason should be remanded, and a final order is entered remanding the action or proceeding to the State court, the defendant or defendants, or any other party who would have been permitted or required to file a pleading  had the proceedings to remove not been instituted, shall have 30 days after the filing in such State court of a certified copy of the order of remand to file motions and to answer or otherwise plead. (b) How Presented. - Every defense, in law or fact, to a claim for relief in any pleading, whether a claim, counterclaim, crossclaim, or third-party claim, shall be asserted in the responsive pleading thereto if one is required, except that the following defenses may at  the option of the pleader be made by motion: (1) Lack of jurisdiction over the subject matter, (2) Lack of jurisdiction over the person, (3) Improper venue or division, (4) Insufficiency of process, (5) Insufficiency of service of process, (6) Failure to state a claim upon which relief can be granted, (7) Failure to join a necessary party. A motion making any of these defenses shall be made before pleading if a further pleading is permitted. The consequences of failure to make such a motion shall be as provided in sections (g) and (h). No defense or objection is waived by being joined with one or more other defenses or objections in a responsive pleading or motion. Obtaining an extension of time within which to answer or otherwise plead shall not constitute a waiver of any defense herein set forth. If a pleading sets forth a claim for relief to which the adverse party is not required to serve a responsive pleading, he may assert at the trial any defense in law or fact to that claim for relief. If, on a motion asserting the defense numbered (6), to dismiss for failure of the pleading to state a claim upon which relief can be granted, matters outside the pleading are presented to and not excluded by the court, the motion shall be treated as one for summary judgment and disposed of as provided in Rule 56, and all parties shall be given reasonable opportunity to present all material made pertinent to such a motion by Rule 56. (c) Motion for judgment on the pleadings. - After the pleadings are closed but within such time as not to delay the trial, any party may move for judgment on the pleadings. If, on a motion for  judgment on the pleadings, matters outside the pleadings are presented to and not excluded by the court, the motion shall be treated as one for summary judgment and disposed of as provided in Rule 56, and all parties shall be given reasonable opportunity to present all material made pertinent to such a motion by Rule 56. (d) Preliminary hearings. - The defenses specifically enumerated (1) through (7) in section (b) of this rule, whether made in a pleading or by motion, and the motion for judgment on the pleadings mentioned in section (c) of this rule shall be heard and determined before trial on application of any party, unless the judge orders that the hearing and determination thereof be deferred until the trial. (e) Motion for more definite statement. - If a pleading to which a responsive pleading is permitted is so vague or ambiguous that a party cannot reasonably be required to frame a responsive pleading, he may move for a more definite statement before interposing his responsive pleading. The motion shall point out the defects complained of and the details desired. If the motion is granted and the order of the judge is not obeyed within 20 days after notice of the order or within such other time as the judge may fix, the judge may strike the pleading to which the motion was directed or make such orders as he deems just. (f) Motion to strike. - Upon motion made by a party before responding to a pleading or, if no responsive pleading is permitted by these rules, upon motion made by a party within 30 days after the service of the pleading upon him or upon the judge's own initiative at any time, the judge may order stricken from any pleading any insufficient defense or any redundant, irrelevant, immaterial, impertinent, or scandalous matter. (g) Consolidation of defenses in motion. - A party who makes a motion under this rule may join with it any other motions herein provided for and then available to him. If a party makes a motion under this rule but omits therefrom any defense or objection then available to him which this rule permits to be raised by motion, he shall not thereafter make a motion based on the defense or objection so omitted, except a motion as provided in section (h)(2) hereof on any of the grounds there stated. (h) Waiver or preservation of certain defenses. - (1) A defense of lack of jurisdiction over the person, improper venue, insufficiency of process, or insufficiency of service  of process is waived (i) if omitted from a motion in the circumstances described in section (g), or (ii) if it is neither made by motion under this rule nor included in a responsive pleading or an amendment thereof permitted by Rule 15(a) to be made as a matter of course. (2) A defense of failure to state a claim upon which relief can be granted, a defense of failure to join a necessary party, and an objection of failure to state a legal defense to a claim may be made in any pleading permitted or ordered under  Rule 7(a), or by motion for judgment on the pleadings, or at the trial on the merits. (3) Whenever it appears by suggestion of the parties or otherwise that the court lacks jurisdiction of the subject matter, the court shall dismiss the action. (1967, c. 954, s. 1; 1971, c. 1236; 1975, c. 76, s. 2.) Rule 13. Counterclaim and crossclaim. (a) Compulsory counterclaims. - A pleading shall state as a counterclaim any claim which at the time of serving the pleading the pleader has against any opposing party, if it arises out of the transaction or occurrence that is the subject matter of the opposing party's claim and does not require for its adjudication the presence of third parties of whom the court cannot acquire jurisdiction. But the pleader need not state the claim if (1) At the time the action was commenced the claim was the subject of another pending action, or (2) The opposing party brought suit upon his claim by attachment or other process by which the court did not acquire jurisdiction to render a personal judgment on that claim, and the pleader is not stating any counterclaim under this rule. (b) Permissive counterclaim. - A pleading may state as a counterclaim any claim against an opposing party not arising out of the transaction or occurrence that is the subject matter of the opposing party's claim. (c) Counterclaim exceeding opposing claim. - A counterclaim may or may not diminish or defeat the recovery sought by the opposing party. It may claim relief exceeding in amount or different in kind from that sought in the pleading of the opposing party. (d) Counterclaim against the State of North Carolina. - These rules shall not be construed to enlarge beyond the limits fixed by law the right to assert counterclaims or to claim credit against the State of North Carolina or an officer or agency thereof. (e) Counterclaim maturing or acquired after pleading. - A claim which either matured or was acquired by the pleader after serving his pleading may, with the permission of the court, be presented as a counterclaim by supplemental pleading. (f) Omitted counterclaim. - When a pleader fails to set up a counterclaim through oversight, inadvertence, or excusable neglect, or when justice requires, he may by leave of court set up the counterclaim by amendment. (g) Crossclaim against coparty. - A pleading may state as a crossclaim any claim by one party against a coparty arising out of the transaction or occurrence that is the subject matter either of the original action or of a counterclaim therein or relating to any property that is the subject matter of the original action. Such crossclaim may include a claim that the party against whom it is asserted is or may be liable to the crossclaimant for all or part of a claim asserted in the action against the crossclaimant. (h) Additional parties may be brought in. - When the presence of parties other than those to the original action is required for the granting of complete relief in the determination of a counterclaim or  crossclaim, the court shall order them to be brought in as defendants  as provided in these rules, if jurisdiction of them can be obtained. (i) Separate trial; separate judgment. - If the court orders separate trials as provided in Rule 42(b), judgment on a counterclaim or crossclaim may be rendered in accordance with the terms of Rule 54(b) when the court has jurisdiction so to do, even if the claims of  the opposing party have been dismissed or otherwise disposed of. (1967, c. 954, s. 1.)"""

# clauses = extract_cited_clauses(text)
# for clause in clauses:
#     print(f"\nClause: {clause['clause']}")
#     print(f"Citations: {clause['citations']}")

# print(longest_disjoint_common_substrings(text, context))
