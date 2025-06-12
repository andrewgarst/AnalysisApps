import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Bar, Line } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

export default function CostModelApp() {
  const [inputs, setInputs] = useState({
    numVariants: 192,
    pctCorrect: 0.84,
    roundLength: 4.4,
    delayWeeks: 1.1,
    costPerVariant: 30.0,
    costSeq: 5000.0,
    fteHours: 60,
    fteCostPerHour: 125,
  });

  const computeCosts = () => {
    const {
      numVariants,
      pctCorrect,
      roundLength,
      delayWeeks,
      costPerVariant,
      costSeq,
      fteHours,
      fteCostPerHour,
    } = inputs;

    const costSynth = numVariants * costPerVariant;
    const costLost = numVariants * (1 - pctCorrect) * costPerVariant;
    const costDelay = delayWeeks * 5 * 8 * fteCostPerHour;
    const costResearcher = fteHours * fteCostPerHour;
    const total = costSynth + costLost + costDelay + costResearcher + costSeq;

    return { costSynth, costLost, costDelay, costResearcher, costSeq, total };
  };

  const costs = computeCosts();

  const chartData = {
    labels: ["Synthesis", "Lost Variants", "Delay", "Researcher", "Sequencing"],
    datasets: [
      {
        label: "Cost Breakdown ($)",
        data: [
          costs.costSynth,
          costs.costLost,
          costs.costDelay,
          costs.costResearcher,
          costs.costSeq,
        ],
        backgroundColor: ["#3F7F6F", "#56C1AC", "#FF4B4B", "#00C853", "#10282D"],
      },
    ],
  };

  return (
    <div className="p-4 max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-center">Screening Cost Model</h1>
      <Tabs defaultValue="inputs" className="w-full">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="inputs">Inputs</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        <TabsContent value="inputs">
          <Card>
            <CardContent className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
              {Object.keys(inputs).map((key) => (
                <div key={key}>
                  <label className="text-sm font-medium capitalize">{key.replace(/([A-Z])/g, " $1")}</label>
                  <Input
                    type="number"
                    value={inputs[key]}
                    onChange={(e) =>
                      setInputs({ ...inputs, [key]: parseFloat(e.target.value) })
                    }
                  />
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results">
          <Card>
            <CardContent className="pt-6">
              <h2 className="text-xl font-semibold">Total Cost: ${costs.total.toFixed(2)}</h2>
              <Bar data={chartData} options={{ responsive: true }} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
