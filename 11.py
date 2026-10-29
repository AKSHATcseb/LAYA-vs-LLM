import laya

agent = laya.load("convaiinnovations/laya")

out = agent.system_one(
"Requirement: The system shall be fast.",
{
"ambiguity": {
"type": "noul",
"instructions": "Is this requirement ambiguous?"
}
},
)

print(out)
