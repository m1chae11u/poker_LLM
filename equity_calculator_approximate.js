const equityUtils = require('./openpokertools.com/src/utils/equity_utils');

// meant to simulate pre-flop equity only
const approximateEquity = (heroHand, villainRange, numSimulations) => {
  try {
    const [winPercentage, tiePercentage] = equityUtils.approximateHandRangeEquity(
      heroHand, 
      villainRange, 
      numSimulations
    );

    console.log(JSON.stringify({
      win: winPercentage,
      tie: tiePercentage
    }));
  } catch (err) {
    console.error("Error calculating approximate equity:", err);
  }
};

const args = process.argv.slice(2);
const heroHand = JSON.parse(args[0]);
const villainRange = JSON.parse(args[1]);
const numSimulations = parseInt(args[2], 10);  

approximateEquity(heroHand, villainRange, numSimulations);